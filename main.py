import os
import re
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Request
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

origins = [
    "https://ai-resume-pro-ten.vercel.app",
    "ai-resume-pro-nikhils-projects-eb3e72b0.vercel.app",
    "https://ai-resume-builder-service-66nz.onrender.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS_RESUME = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/r1t2-chimera:free",
    "mistralai/devstral2-2512:free",
    "mistralai/devstral2-2512:free",
]

FREE_MODEL_ATS = "tngtech/deepseek-r1t2-chimera:free"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 120

def build_prompt(profile: UserProfile, job: JobDescription) -> str:
    return f"""
            You are an expert resume writing assistant.

            Your task: Generate a tailored resume that follows the schema below and is optimized for ATS (assured 90%+ keyword match and 90+ ATS score).

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
  "score": int (0-100),
  "keywordMatch": int (0-100),
  "missingKeywords": [string],
  "recommendations": [string],
  "formatCompliance": int (0-100),
  "details": object
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
        if isinstance(value, float) and 0 <= value <= 1:
            return int(round(value * 100))
        return int(round(float(value)))
    except Exception:
        return 0

def normalize_ats_result(raw: dict) -> dict:
    score = raw.get("score")
    keywordMatch = raw.get("keywordMatch")
    formatCompliance = raw.get("formatCompliance")
    missing = raw.get("missingKeywords") or []
    recs = raw.get("recommendations") or []

    return {
        "score": _pct_of(score),
        "keywordMatch": _pct_of(keywordMatch),
        "missingKeywords": missing if isinstance(missing, list) else [],
        "recommendations": recs if isinstance(recs, list) else [],
        "formatCompliance": _pct_of(formatCompliance),
        "details": raw,
    }

def extract_json(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Empty response")

    raw = raw.strip()

    fenced = re.search(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
        raw,
        re.IGNORECASE,
    )
    if fenced:
        return json.loads(fenced.group(1))

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(raw[first:last + 1])

    raise ValueError("No JSON found")


async def call_openrouter(prompt: str, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "ATS Resume Backend")
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return extract_json(content)

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
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

@app.post("/generate-resume")
async def generate_resume(payload: GenerateResumePayload):
    prompt = build_prompt(payload.profile, payload.jobDescription)
    models = [payload.modelOverride] if payload.modelOverride else FREE_MODELS_RESUME
    resume = await generate_with_fallback(prompt, models)
    return {"resume": resume}

@app.post("/generate-ats")
async def generate_ats(payload: GenerateAtsPayload):
    prompt = build_prompt_for_ats(payload.resume, payload.jobDescription)
    raw = await generate_with_fallback(prompt, [FREE_MODEL_ATS])
    normalized = normalize_ats_result(raw)
    return JSONResponse(content=normalized)

@app.get("/")
def root():
    return {"status": "AI resume backend running (OpenRouter free models + retries)"}
