import os
import re
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from google import genai
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
    aiProvider: Optional[str] = "Gemini"
    modelOverride: Optional[str] = None

class ResumeInput(BaseModel):
    id: Optional[str] = None
    content: Union[str, Dict[str, Any]]

class GenerateAtsPayload(BaseModel):
    resume: ResumeInput
    jobDescription: JobDescription
    modelOverride: Optional[str] = None

app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MISTRAL_MODEL = os.getenv("MISTRAL_MODEL_ID", "mistralai/mistral-7b-instruct:free")
DEFAULT_LLAMA_MODEL = os.getenv("LLAMA_MODEL_ID", "meta-llama/llama-3.1-405b-instruct:free")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_ID", "deepseek/deepseek-chat-v3.1:free")
DEFAULT_DEEPSEEK_MODEL_ATS = os.getenv("DEEPSEEK_ATS_MODEL_ID", "deepseek/deepseek-r1-distill-llama-70b:free")

def build_prompt(profile: UserProfile, job: JobDescription) -> str:
    return f"""
            You are an expert resume writing assistant.

            Your task: Generate a tailored resume that follows the schema below and is optimized for ATS (>85% keyword match and >90 ATS score).

            STRICT INSTRUCTIONS:
            - All work experience bullets must follow the format: "Did X to achieve Y using Z".
            - Adapt projects and achievements to highlight what’s most relevant to the job description.
            - If a required skill/experience is missing:
            • Expand responsibilities in real past roles (if realistic).
            • Or enhance/modify existing projects to include that technology.
            • Or invent new, realistic side projects (aligned with backend/distributed systems engineering).
            - Do NOT fabricate fake companies or job titles.
            - Use impact-driven language, not JD copy-paste.
            - Ensure formatting is ATS-friendly (plain text, no tables/columns/images).
            - Must fit within 1 page equivalent.
            - Include a strong, tailored Professional Summary.

            Output MUST be valid JSON and follow this schema:

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
                "startDate": "YYYY-MM",
                "endDate": "YYYY-MM or 'Present'",
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
                "graduationDate": "YYYY-MM"
                }}
            ],
            "certifications": ["string"]
            }}

            Now rewrite the resume for ATS optimization using the following data:

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

    return f"""You are an Applicant Tracking System evaluator. Assess how well the resume matches the job.
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
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    text = fenced.group(1) if fenced else raw
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="AI did not return JSON")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {e}")

async def generate_gemini(prompt: str):
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={"temperature": 0.7, "response_mime_type": "application/json"}
    )
    return extract_json(resp.text.strip())

async def generate_openrouter(prompt: str, model_id: str):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "ATS Resume Backend")
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are an expert ATS resume writer. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"OpenRouter error: {r.status_code} {r.text}")
        data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise HTTPException(status_code=500, detail="Empty response from OpenRouter")
    return extract_json(content)

async def generate_mistral(prompt: str, model_override: Optional[str] = None):
    model = model_override or DEFAULT_MISTRAL_MODEL
    return await generate_openrouter(prompt, model)

async def generate_llama(prompt: str, model_override: Optional[str] = None):
    model = model_override or DEFAULT_LLAMA_MODEL
    return await generate_openrouter(prompt, model)

async def generate_deepseek(prompt: str, model_override: Optional[str] = None):
    model = model_override or DEFAULT_DEEPSEEK_MODEL
    return await generate_openrouter(prompt, model)

async def generate_ats_deepseek(prompt: str):
    model = DEFAULT_DEEPSEEK_MODEL_ATS
    return await generate_openrouter(prompt, model)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("RequestValidationError: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

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
    try:
        prompt = build_prompt(payload.profile, payload.jobDescription)
        provider = (payload.aiProvider or "Gemini").strip().lower()
        if provider == "gemini":
            structured_resume = await generate_gemini(prompt)
        elif provider == "mistral":
            structured_resume = await generate_mistral(prompt, payload.modelOverride)
        elif provider == "llama":
            structured_resume = await generate_llama(prompt, payload.modelOverride)
        elif provider == "deepseek":
            structured_resume = await generate_deepseek(prompt, payload.modelOverride)
        else:
            raise HTTPException(status_code=400, detail="Unknown provider")
        return {"resume": structured_resume}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {e}")

@app.get("/")
def read_root():
    return {"status": "AI resume backend is running (Gemini + OpenRouter: Mistral, Llama)!"}

