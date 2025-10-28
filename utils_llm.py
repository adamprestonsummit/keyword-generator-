# utils_llm.py
import os, json
from typing import List
from openai import OpenAI
import google.generativeai as genai
from google.api_core.exceptions import NotFound

EXPAND_SYS = "You are an SEO keyword ideation assistant. Return diverse, de-duplicated keyword ideas only (no numbering)."
# ... (keep the rest of your constants)

def _openai_chat(api_key, system, user, temperature=0.2):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return resp.choices[0].message.content

def _gemini_chat(api_key, system, user, temperature=0.2, model_name=None):
    genai.configure(api_key=api_key)
    # Try a small list of common, account-dependent model aliases
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
                generation_config={"temperature": float(temperature)}
            )
            return getattr(resp, "text", "").strip()
        except NotFound as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    # If we got here, none worked
    raise RuntimeError(
        f"Gemini model not found/available for this API key. Tried: {', '.join(candidates)}. "
        f"Detail: {type(last_err).__name__}: {last_err}"
    )

def llm_expand_keywords(provider, api_key, n_ideas, temperature, seed, source_text, audience, market, language) -> List[str]:
    system = EXPAND_SYS
    user = EXPAND_USER.format(
        seed=seed or "(none)",
        source_text=(source_text[:4000] if source_text else ""),
        n_ideas=n_ideas,
        audience=audience,
        market=market,
        language=language
    )
    text = _openai_chat(api_key, system, user, temperature) if provider=="OpenAI" else _gemini_chat(api_key, system, user, temperature)
    lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        if l[:3].strip(". ").isdigit() and "." in l:
            l = l.split(".", 1)[1].strip()
        cleaned.append(l)
    return list(dict.fromkeys(cleaned))


def llm_tag_batch(provider, api_key, keywords: List[str], market, language):
    out = []
    for i in range(0, len(keywords), 50):
        chunk = keywords[i:i+50]
        system = TAG_SYS
        user = TAG_USER.format(
            market=market, language=language,
            keywords="\n".join(f"- {k}" for k in chunk)
        )
        text = _openai_chat(api_key, system, user, 0.1) if provider=="OpenAI" else _gemini_chat(api_key, system, user, 0.1)
        # Try JSON parse; fall back safely
        try:
            data = json.loads(text)
            if isinstance(data, list):
                out.extend(data)
                continue
        except Exception:
            pass
        for k in chunk:
            out.append({"keyword": k, "intent": "", "funnel": "", "theme": ""})
    return out
