import os
import re
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import httpx
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger("uvicorn.error")
load_dotenv()

class PersonalInfo(BaseModel):
    name: str
    email: str
    phone: str
    location: str
    website: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None

class WorkExperience(BaseModel):
    id: str
    title: str
    company: str
    location: str
    startDate: str
    endDate: str
    current: bool
    achievements: List[str]
    technologies: List[str]

class Project(BaseModel):
    id: str
    name: str
    description: str
    technologies: List[str]
    url: Optional[str] = None
    github: Optional[str] = None
    achievements: List[str]

class Education(BaseModel):
    id: str
    degree: str
    institution: str
    location: str
    graduationDate: str
    gpa: Optional[str] = None
    relevant_courses: Optional[List[str]] = Field(default=None, alias="relevant_courses")

class UserProfile(BaseModel):
    personalInfo: PersonalInfo
    summary: str
    experience: List[WorkExperience]
    projects: List[Project]
    skills: Dict[str, Any]
    education: List[Education]
    certifications: List[str]

class JobDescription(BaseModel):
    title: str = ""
    company: str = ""
    description: str = ""
    requirements: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

class GenerateResumePayload(BaseModel):
    profile: UserProfile
    jobDescription: JobDescription
    modelOverride: Optional[str] = None

class ResumeInput(BaseModel):
    id: Optional[str] = None
    content: Union[str, Dict[str, Any]]

class GenerateAtsPayload(BaseModel):
    resume: ResumeInput
    jobDescription: JobDescription

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-resume-pro-ten.vercel.app",
        "ai-resume-pro-nikhils-projects-eb3e72b0.vercel.app",
        "https://ai-resume-builder-service-66nz.onrender.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS_RESUME = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

FREE_MODEL_ATS = "google/gemma-3-27b-it:free"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 120

def build_prompt(profile: UserProfile, job: JobDescription) -> str:
    return f"""
You are an expert ATS-focused resume writer.

Generate a 1-page ATS-optimized resume.

Rules:
- Bullets must follow: "Did X to achieve Y using Z".
- Prioritize job-relevant skills.
- Do NOT invent companies or titles.
- Plain text only.
- Return ONLY valid JSON matching the schema below.

Schema:
{{
  "summary": "string",
  "skills": {{
    "Languages": ["string"],
    "Frameworks & Tools": ["string"],
    "Databases & Storage": ["string"],
    "Infrastructure & Cloud": ["string"],
    "System Design": ["string"],
    "Performance Optimization": ["string"]
  }},
  "work_experience": [
    {{
      "title": "string",
      "company": "string",
      "location": "string",
      "startDate": "MM-YYYY",
      "endDate": "MM-YYYY or 'Present'",
      "achievements": ["Did X to achieve Y using Z"],
      "technologies": ["string"]
    }}
  ],
  "projects": [
    {{
      "name": "string",
      "description": "string",
      "technologies": ["string"],
      "achievements": ["Did X to achieve Y using Z"]
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "graduationDate": "MM-YYYY"
    }}
  ],
  "certifications": ["string"]
}}

User Profile:
{profile.model_dump_json(indent=2)}

Job Description:
{job.model_dump_json(indent=2)}
"""

def build_prompt_for_ats(resume: Union[str, Dict[str, Any]], job_description: Union[str, Dict[str, Any]]) -> str:
    def to_text(x: Union[str, Dict[str, Any]]) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, dict):
            # unwrap if passed as { "content": {...} }
            if 'content' in x:
                return to_text(x['content'])
            parts = []
            if x.get("summary"):
                parts.append(f"summary: {x.get('summary')}")
            if x.get("skills"):
                parts.append(f"skills: {', '.join([k+':'+','.join(v if isinstance(v, list) else [v]) for k,v in x.get('skills', {}).items()])}")
            if x.get("work_experience"):
                for we in x.get("work_experience", []):
                    title = we.get("title", "")
                    company = we.get("company", "")
                    ach = " ; ".join(we.get("achievements", []))
                    parts.append(f"{title} @ {company} — {ach}")
            if x.get("projects"):
                for p in x.get("projects", []):
                    parts.append(f"project: {p.get('name','')} — {p.get('description','')}")
            return "\n".join(parts).strip()
        return ""

    resume_text = to_text(resume)
    job_text = ""
    if isinstance(job_description, dict) and job_description.get("description"):
        job_text = job_description.get("description", "")
    else:
        job_text = to_text(job_description)

    return f"""You are an Applicant Tracking System (ATS) evaluator.
Compare the following resume against the job description
and return ONLY a valid JSON object in the schema below.

Schema:
{{
  "score": int (0-100),                   # overall ATS score
  "keywordMatch": int (0-100),            # % of required keywords matched
  "missingKeywords": [string],            # list of missing important keywords
  "recommendations": [string],            # textual recommendations
  "formatCompliance": int (0-100),        # compliance with ATS-friendly formatting
  "details": object                       # free-form breakdown (skills, experience, etc.)
}}

RESUME_START
{resume_text}
RESUME_END

JOB_START
{job_text}
JOB_END
"""

def _pct_of(value) -> int:
    if value is None:
        return 0
    try:
        # handle floats like 0.85 (meaning 85%) and ints like 85
        if isinstance(value, float) and 0 <= value <= 1:
            return int(round(value * 100))
        return int(round(float(value)))
    except Exception:
        return 0

def normalize_ats_result(raw: dict) -> dict:
    """
    Normalize many possible LLM output shapes into the canonical ATS schema:
    {
      "score": int(0-100),
      "keywordMatch": int(0-100),
      "missingKeywords": [...],
      "recommendations": [...],
      "formatCompliance": int(0-100),
      "details": { ... }  # original raw payload
    }
    """

    # 1) score candidates
    score = None
    if raw.get("score") is not None:
        score = _pct_of(raw.get("score"))
    elif raw.get("matching_score") is not None:
        score = _pct_of(raw.get("matching_score"))
    elif isinstance(raw.get("details"), dict) and raw["details"].get("matching_score") is not None:
        score = _pct_of(raw["details"]["matching_score"])

    # 2) keywordMatch candidates
    keywordMatch = None
    if raw.get("keywordMatch") is not None:
        keywordMatch = _pct_of(raw.get("keywordMatch"))
    else:
        # try breakdown -> keywords -> confidence / match
        breakdown = raw.get("breakdown") or raw.get("details", {}).get("breakdown") or {}
        kw = breakdown.get("keywords") if isinstance(breakdown, dict) else None
        if isinstance(kw, dict):
            if kw.get("confidence") is not None:
                # confidence may be 0-1 float or 0-100 number
                val = kw.get("confidence")
                keywordMatch = _pct_of(val if not (isinstance(val, float) and 0 <= val <= 1) else val * 100)
            elif kw.get("match") is not None:
                keywordMatch = _pct_of(kw.get("match"))
        # fallback: try details.matchedKeywords vs totalKeywords
        if keywordMatch is None:
            matched = raw.get("details", {}).get("matchedKeywords")
            total = raw.get("details", {}).get("totalKeywords")
            if isinstance(matched, list) and isinstance(total, int) and total > 0:
                keywordMatch = _pct_of((len(matched) / total) * 100)

    # 3) missingKeywords
    missing = raw.get("missingKeywords")
    if missing is None:
        missing = raw.get("details", {}).get("missingKeywords") or raw.get("details", {}).get("highImpactGaps") or []
    if not isinstance(missing, list):
        missing = []

    # 4) recommendations
    recs = raw.get("recommendations") or raw.get("details", {}).get("recommendations") or raw.get("advice") or []
    if not isinstance(recs, list):
        recs = [str(recs)] if recs else []

    # 5) formatCompliance
    formatComp = None
    if raw.get("formatCompliance") is not None:
        formatComp = _pct_of(raw.get("formatCompliance"))
    elif raw.get("format_score") is not None:
        formatComp = _pct_of(raw.get("format_score"))
    elif isinstance(raw.get("details"), dict) and raw["details"].get("formatCompliance") is not None:
        formatComp = _pct_of(raw["details"]["formatCompliance"])

    # final normalization with sensible defaults
    normalized = {
        "score": score if score is not None else 0,
        "keywordMatch": keywordMatch if keywordMatch is not None else 0,
        "missingKeywords": missing,
        "recommendations": recs,
        "formatCompliance": formatComp if formatComp is not None else 0,
        "details": raw,
    }
    return normalized

def extract_json(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty response")
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
    text = fenced.group(1) if fenced else raw
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON found")
    return json.loads(text[first:last + 1])

async def call_openrouter(prompt: str, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return extract_json(data["choices"][0]["message"]["content"])

async def generate_with_fallback(prompt: str, models: List[str]) -> dict:
    last_error = None
    for _ in range(MAX_RETRIES):
        for model in models:
            try:
                return await call_openrouter(prompt, model)
            except Exception as e:
                last_error = e
                await asyncio.sleep(0.5)
    raise HTTPException(status_code=500, detail=str(last_error))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.post("/generate-ats")
async def generate_ats(payload: GenerateAtsPayload):
    try:
        prompt = build_prompt_for_ats(payload.resume, payload.jobDescription)
        raw_result = await generate_ats_deepseek(prompt)

        # raw_result should be a dict parsed from the LLM response. Normalize for frontend.
        normalized = normalize_ats_result(raw_result)
        return JSONResponse(content=normalized)
    except HTTPException:
        # re-raise known HTTPExceptions
        raise
    except Exception as e:
        logger.exception("generate_ats failed: %s", e)
        # return a 500 with a message (frontend will receive JSON)
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {e}")


@app.post("/generate-resume")
async def generate_resume(payload: GenerateResumePayload):
    prompt = build_prompt(payload.profile, payload.jobDescription)
    models = [payload.modelOverride] if payload.modelOverride else FREE_MODELS_RESUME
    resume = await generate_with_fallback(prompt, models)
    return {"resume": resume}

@app.get("/")
def root():
    return {"status": "AI resume backend running"}
