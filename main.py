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
    condensed: Optional[bool] = False

class ResumeInput(BaseModel):
    id: Optional[str] = None
    content: Union[str, Dict[str, Any]]

class GenerateAtsPayload(BaseModel):
    resume: ResumeInput
    jobDescription: JobDescription
    modelOverride: Optional[str] = None

app = FastAPI()


origins = [
    "https://ai-resume-pro-ten.vercel.app",
    "https://ai-resume-pro-nikhils-projects-eb3e72b0.vercel.app",
    "https://ai-resume-builder-service-66nz.onrender.com",  # optional if your frontend fetches from same origin
    "http://localhost:3000",  # local dev
    "http://localhost:5173",  # vite dev
    "https://ai-resume-backend-97113263099.asia-south1.run.app",  # cloud run
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # or ["*"] for testing
    allow_credentials=True,        # set True only if you send cookies/auth
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_PRIMARY_MODEL = "gemini-3-flash-preview"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MISTRAL_MODEL = os.getenv("MISTRAL_MODEL_ID", "nvidia/nemotron-3-nano-30b-a3b:free")
DEFAULT_LLAMA_MODEL = os.getenv("LLAMA_MODEL_ID", "meta-llama/llama-3.3-70b-instruct:free")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_ID", "qwen/qwen3-coder:free")
DEFAULT_DEEPSEEK_MODEL_ATS = os.getenv("DEEPSEEK_ATS_MODEL_ID", "nvidia/nemotron-3-nano-30b-a3b:free")

def build_prompt(profile: UserProfile, job: JobDescription, condensed: bool = False) -> str:
    condensed_block = ""
    if condensed:
        condensed_block = """
            CRITICAL — COMPACT MODE (previous output was too long for 1 page):
            REDUCE content length while keeping the page FULL and DENSE:
            - Summary: 2 sentences max.
            - Skills: 5 categories max, shorter lists.
            - Work experience: 4 bullets for current role, 3 for older. Each bullet 15-22 words max.
            - Projects: exactly 3 projects, 2 bullets each.
            - Education: 1 line.
            - Keep the bold-prefix format on bullets. Keep metrics. Just tighten language.
"""

    return f"""
            You are an expert resume writing assistant.

            Your task: Generate a tailored resume that follows the schema below and is optimized for ATS (assured 90%+ keyword match and 90+ ATS score).

            STRICT INSTRUCTIONS:
            - Adapt projects and achievements to highlight what’s most relevant to the job description.
            - If a required skill/experience is missing:
            • Expand responsibilities in real past roles (if realistic).
            • Or enhance/modify existing projects to include that technology.
            • Or invent new, realistic side projects (aligned with the user’s domain).
            - Do NOT fabricate fake companies or job titles.
            - Do NOT invent or hallucinate project URLs. Always set project "url" to null. Only the user can provide real URLs.
            - Use impact-driven language with quantified metrics (%, x, K, M numbers).
            - Ensure formatting is ATS-friendly (plain text, no tables/columns/images).

            ACTION VERB RULE (CRITICAL — NO REPEATED VERBS):
            - Every bullet across the ENTIRE resume must start with a UNIQUE action verb after the prefix.
            - NEVER reuse verbs like: Built, Developed, Implemented, Designed, Engineered, Created, Owned, Led, Managed.
            - Use each verb ONLY ONCE across all bullets. Vary aggressively. Examples of diverse verbs:
              Built, Architected, Spearheaded, Pioneered, Orchestrated, Optimized, Reduced, Streamlined,
              Shipped, Delivered, Automated, Migrated, Converted, Scaled, Drove, Eliminated, Accelerated,
              Introduced, Revamped, Transformed, Consolidated, Integrated, Overhauled, Established.
            - Before finalizing, scan all bullets and replace any duplicated verbs.

            BULLET FORMAT (CRITICAL — every experience bullet MUST follow this pattern):
            "Bold Prefix Title: UniqueVerb rest of description with metrics"
            Example: "Distributed Caching System: Architected a gRPC-based distributed cache with pluggable modules, reducing cache misses by 35% and improving read throughput by 70%."
            The prefix before the colon should be a short 2-4 word topic label. The description after should be 1-2 lines with metrics.

            PAGE LENGTH CONSTRAINTS (CRITICAL — must fill exactly 1 printed page, not more, not less):
            - Summary: 2-3 sentences. Write in a natural, humanized tone — not robotic or keyword-stuffed. Sound like a real person describing their strengths conversationally but professionally.
            - Skills: 5-7 categories (e.g., Languages, Frameworks & APIs, Databases & Storage, Infrastructure & Cloud, AI/ML, System Design). Each category has comma-separated values.
            - Work experience: 5-6 bullet points for the most recent/current role. 3-4 bullets for older roles. Each bullet 15-30 words.
            - Projects: exactly 3 projects, each with exactly 2 bullet points. No description field needed — put everything in achievements.
            - Education: 1 line per entry (institution, location, degree, date).
            - The page must be FULL — dense with keywords, metrics, and impact. No wasted space. No half-empty page.
            {condensed_block}

            Output MUST be valid JSON and follow this schema:

            {{
            "summary": "string (2-3 sentences)",
            "skills": {{
                "Languages": ["string"],
                "Frameworks & APIs": ["string"],
                "Caching & Messaging": ["string"],
                "Databases & Storage": ["string"],
                "AI / ML Systems": ["string"],
                "Infrastructure & Cloud": ["string"],
                "System Design": ["string"]
            }},
            "work_experience": [
                {{
                "title": "string",
                "company": "string",
                "location": "string (city)",
                "startDate": "MM/YYYY",
                "endDate": "MM/YYYY or 'Present'",
                "achievements": ["Bold Prefix: Description with metrics (each bullet 15-30 words)"]
                }}
            ],
            "projects": [
                {{
                "name": "string",
                "url": null,
                "achievements": ["string (each bullet 15-25 words)"]
                }}
            ],
            "education": [
                {{
                "degree": "string",
                "institution": "string",
                "location": "string (city)",
                "graduationDate": "MM/YYYY - MM/YYYY"
                }}
            ]
            }}

            Now rewrite the resume for ATS optimization using the following data:

            User Profile:
            {profile.model_dump_json()}

            Job Description:
            {job.model_dump_json()}
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
                    ach = " ; ".join(p.get("achievements", []))
                    parts.append(f"project: {p.get('name','')} — {p.get('description', '')} {ach}")
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

async def _call_gemini(prompt: str, model: str):
    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json"
        }
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        if r.status_code >= 400:
            raise Exception(f"{r.status_code} {r.text[:200]}")
        data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json(text.strip())

async def generate_gemini(prompt: str, max_retries: int = 3):
    # Try primary model (gemini-3-flash-preview)
    for attempt in range(max_retries):
        try:
            return await _call_gemini(prompt, GEMINI_PRIMARY_MODEL)
        except Exception as e:
            err_str = str(e)
            if ("503" in err_str or "UNAVAILABLE" in err_str or "overload" in err_str.lower() or "429" in err_str) and attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                logger.warning(f"Gemini primary retry {attempt+1}/{max_retries} after {wait}s")
                await asyncio.sleep(wait)
                continue
            break
    # Fallback to 2.5-flash
    logger.warning(f"Gemini primary ({GEMINI_PRIMARY_MODEL}) failed, falling back to {GEMINI_FALLBACK_MODEL}")
    return await _call_gemini(prompt, GEMINI_FALLBACK_MODEL)

async def generate_openrouter(prompt: str, model_id: str, max_retries: int = 2):
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
    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
            if r.status_code == 429 and attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                logger.warning(f"OpenRouter 429 retry {attempt+1}/{max_retries} for {model_id} after {wait}s")
                await asyncio.sleep(wait)
                continue
            if r.status_code >= 400:
                raise HTTPException(status_code=500, detail=f"OpenRouter error: {r.status_code} {r.text}")
            data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise HTTPException(status_code=500, detail="Empty response from OpenRouter")
        return extract_json(content)
    raise HTTPException(status_code=500, detail=f"OpenRouter failed after {max_retries} retries")

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
        resume_data = payload.resume.content if hasattr(payload.resume, 'content') else payload.resume
        job_data = payload.jobDescription.model_dump() if hasattr(payload.jobDescription, 'model_dump') else payload.jobDescription
        prompt = build_prompt_for_ats(resume_data, job_data)
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
        prompt = build_prompt(payload.profile, payload.jobDescription, condensed=bool(payload.condensed))
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

class ExtractKeywordsPayload(BaseModel):
    description: str

@app.post("/extract-keywords")
async def extract_keywords(payload: ExtractKeywordsPayload):
    try:
        prompt = f"""Extract the most important technical keywords, skills, and requirements from this job description.
Return ONLY a JSON object with this schema:
{{
  "keywords": ["string"],
  "nice_to_have": ["string"]
}}
"keywords" = must-have skills/technologies mentioned.
"nice_to_have" = preferred/bonus skills.
Keep each item short (1-3 words). Max 15 keywords, max 8 nice_to_have.

JOB DESCRIPTION:
{payload.description}
"""
        result = await generate_gemini(prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword extraction failed: {e}")

class CoverLetterPayload(BaseModel):
    resume: Union[str, Dict[str, Any]]
    jobDescription: JobDescription
    companyName: Optional[str] = None

@app.post("/generate-cover-letter")
async def generate_cover_letter(payload: CoverLetterPayload):
    try:
        resume_text = payload.resume if isinstance(payload.resume, str) else json.dumps(payload.resume, indent=2)
        company = payload.companyName or payload.jobDescription.company or "the company"
        prompt = f"""You are an expert cover letter writer.

Write a professional, compelling cover letter for the following job based on the candidate's resume.

RULES:
- 3-4 paragraphs, ~250-350 words total.
- Opening: mention the specific role and company, show genuine interest.
- Body: highlight 2-3 most relevant achievements from the resume that directly match the job requirements. Use specific metrics.
- Closing: express enthusiasm, mention availability, call to action.
- Tone: confident but not arrogant, professional but personable.
- Do NOT repeat the resume verbatim — reframe achievements in narrative form.
- Do NOT use generic filler like "I am writing to express my interest" — start with something specific.

Return ONLY a JSON object:
{{
  "subject": "string (email subject line)",
  "body": "string (the full cover letter text with \\n for line breaks)"
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
Title: {payload.jobDescription.title}
Company: {company}
Description: {payload.jobDescription.description}
"""
        # Try DeepSeek first (2 attempts), fallback to Gemini
        for attempt in range(2):
            try:
                result = await generate_deepseek(prompt)
                return result
            except Exception:
                if attempt < 1:
                    await asyncio.sleep(3)
                    continue
        # DeepSeek failed twice, fallback to Gemini
        result = await generate_gemini(prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cover letter generation failed: {e}")

@app.get("/")
def read_root():
    return {"status": "AI resume backend is running (Gemini + OpenRouter: Mistral, Llama)!"}
