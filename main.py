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
    "deepseek/r1t2-chimera:free",
    "mistralai/devstral2-2512:free",
    "qwen/qwen2.5-72b-instruct:free",
]

FREE_MODEL_ATS = "deepseek/r1t2-chimera:free"

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

@app.post("/generate-resume")
async def generate_resume(payload: GenerateResumePayload):
    prompt = build_prompt(payload.profile, payload.jobDescription)
    models = [payload.modelOverride] if payload.modelOverride else FREE_MODELS_RESUME
    resume = await generate_with_fallback(prompt, models)
    return {"resume": resume}

@app.get("/")
def root():
    return {"status": "AI resume backend running"}
