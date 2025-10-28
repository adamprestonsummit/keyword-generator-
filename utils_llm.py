import os, json
from typing import List
from openai import OpenAI
import google.generativeai as genai
from google.api_core.exceptions import NotFound

# ---------- Prompt templates ----------
EXPAND_SYS = (
    "You are an SEO keyword ideation assistant. "
    "Return diverse, de-duplicated keyword ideas only (no numbering)."
)

EXPAND_USER = """
Seed: {seed}
Audience: {audience}
Market/Locale: {market}
Language: {language}
If source context is provided, use it:
SOURCE:
{source_text}

Return ~{n_ideas} concise keyword ideas (1 per line), mixing head & long-tail, across intents.
ONLY return the keywords, one per line.
"""

TAG_SYS = (
    "You label SEO keywords with intent (Informational/Commercial/Transactional/Navigational), "
    "funnel (TOFU/MOFU/BOFU), and theme (2-3 words)."
)

TAG_USER = """
Market: {market} | Language: {language}
For each keyword, return JSON array of objects with fields: keyword, intent, funnel, theme.
Keywords:
{keywords}
"""

# ---------- Providers ----------
def _openai_chat(api_key: str, system: str, user: str, temperature: float = 0.2) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=float(temperature),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""

def _gemini_chat(
    api_key: str, system: str, user: str, temperature: float = 0.2, model_name: str | None = None
) -> str:
    genai.configure(api_key=api_key)
    # Try a few common aliases; allow override via env var
    candidates = [
        model_name or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
    ]
    last_err = None
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(
                [system, user],
                generation_config={"temperature": float(temperature)},
            )
            return (getattr(resp, "text", "") or "").strip()
        except NotFound
