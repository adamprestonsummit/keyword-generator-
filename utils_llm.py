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
        "gemini-2.5-flash-latest",
        "gemini-2.5-pro",
        "gemini-2.5-pro-latest",
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
        except NotFound as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Gemini model not available for this key. Tried: {', '.join(candidates)}. "
        f"Detail: {type(last_err).__name__}: {last_err}"
    )

# ---------- Public helpers ----------
def llm_expand_keywords(
    provider: str,
    api_key: str,
    n_ideas: int,
    temperature: float,
    seed: str | None,
    source_text: str | None,
    audience: str,
    market: str,
    language: str,
) -> List[str]:
    system = EXPAND_SYS
    user = EXPAND_USER.format(
        seed=seed or "(none)",
        source_text=(source_text[:4000] if source_text else ""),
        n_ideas=n_ideas,
        audience=audience,
        market=market,
        language=language,
    )
    text = (
        _openai_chat(api_key, system, user, temperature)
        if provider == "OpenAI"
        else _gemini_chat(api_key, system, user, temperature)
    )

    # Clean into a list of keywords
    lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        # remove numbering like "1. foo"
        if l[:3].strip(". ").isdigit() and "." in l:
            l = l.split(".", 1)[1].strip()
        cleaned.append(l)
    # ordered de-dupe
    return list(dict.fromkeys(cleaned))

def llm_tag_batch(
    provider: str, api_key: str, keywords: List[str], market: str, language: str
) -> List[dict]:
    out: List[dict] = []
    for i in range(0, len(keywords), 50):
        chunk = keywords[i : i + 50]
        system = TAG_SYS
        user = TAG_USER.format(
            market=market, language=language, keywords="\n".join(f"- {k}" for k in chunk)
        )
        text = (
            _openai_chat(api_key, system, user, 0.1)
            if provider == "OpenAI"
            else _gemini_chat(api_key, system, user, 0.1)
        )
        # Try JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                # Normalize records
                for rec in data:
                    k = (rec.get("keyword") or "").strip()
                    if not k:
                        continue
                    out.append(
                        {
                            "keyword": k,
                            "intent": rec.get("intent", ""),
                            "funnel": rec.get("funnel", ""),
                            "theme": rec.get("theme", ""),
                        }
                    )
                continue
        except Exception:
            pass
        # Fallback: emit empty tags
        for k in chunk:
            out.append({"keyword": k, "intent": "", "funnel": "", "theme": ""})
    return out
