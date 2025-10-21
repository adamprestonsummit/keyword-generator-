import os, json
from typing import List
from openai import OpenAI
import google.generativeai as genai

EXPAND_SYS = "You are an SEO keyword ideation assistant. Return diverse, de-duplicated keyword ideas only (no numbering)."
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

TAG_SYS = "You label SEO keywords with intent (Informational/Commercial/Transactional/Navigational), funnel (TOFU/MOFU/BOFU), and theme (2-3 words)."
TAG_USER = """
Market: {market} | Language: {language}
For each keyword, return JSON array of objects with fields: keyword, intent, funnel, theme.
Keywords:
{keywords}
"""

def _openai_chat(api_key, system, user, temperature=0.2):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return resp.choices[0].message.content

def _gemini_chat(api_key, system, user, temperature=0.2):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content([system, user], generation_config={"temperature":temperature})
    return resp.text

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
    # remove numbering if present
    cleaned = []
    for l in lines:
        if l[:3].strip(". ").isdigit() and "." in l:
            l = l.split(".", 1)[1].strip()
        cleaned.append(l)
    # ordered de-dupe
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
