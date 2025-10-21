import csv, io, time, requests
from typing import Iterable, Dict, List

SEM_BASE = "https://api.semrush.com/"

MARKET_TO_DB = {
    "United Kingdom": "uk", "UK": "uk",
    "United States": "us", "USA": "us",
    "Germany": "de", "France": "fr", "Spain": "es", "Italy": "it",
    "Australia": "au", "Canada": "ca", "Netherlands": "nl", "Ireland": "ie"
}

def db_for_market(market: str, fallback: str = "uk") -> str:
    return MARKET_TO_DB.get(market, fallback)

def _fetch_csv(params: Dict[str,str], timeout=25) -> List[Dict[str,str]]:
    r = requests.get(SEM_BASE, params=params, timeout=timeout)
    r.raise_for_status()
    s = io.StringIO(r.content.decode("utf-8", errors="replace"))
    return list(csv.DictReader(s))

def _phrase_all(api_key: str, phrase: str, db: str):
    params = {"key": api_key, "type": "phrase_all", "export_columns": "Ph,Nq,Cp,Cc", "phrase": phrase, "database": db}
    rows = _fetch_csv(params)
    return rows[0] if rows else {}

def _phrase_kdi(api_key: str, phrase: str, db: str):
    params = {"key": api_key, "type": "phrase_kdi", "export_columns": "Ph,Kd", "phrase": phrase, "database": db}
    rows = _fetch_csv(params)
    return rows[0] if rows else {}

def _phrase_trends(api_key: str, phrase: str, db: str):
    params = {"key": api_key, "type": "phrase_trends", "export_columns": "Ph,Td", "phrase": phrase, "database": db}
    rows = _fetch_csv(params)
    return rows[0] if rows else {}

def get_semrush_metrics(api_key: str, keywords: Iterable[str], market: str, per_call_sleep: float = 0.2, with_trend: bool = True) -> Dict[str, Dict]:
    db = db_for_market(market)
    out: Dict[str, Dict] = {}
    for kw in keywords:
        kw_norm = kw.strip()
        if not kw_norm:
            continue
        try:
            a = _phrase_all(api_key, kw_norm, db); time.sleep(per_call_sleep)
            k = _phrase_kdi(api_key, kw_norm, db)
            trend = ""
            if with_trend:
                time.sleep(per_call_sleep)
                t = _phrase_trends(api_key, kw_norm, db)
                trend = (t.get("Td") or "").strip()
            out[kw_norm] = {
                "volume": _safe_int(a.get("Nq")),
                "kd": _safe_float(k.get("Kd")),
                "cpc": _safe_float(a.get("Cp")),
                "currency": (a.get("Cc") or "").upper(),
                "trend": trend,
            }
        except Exception:
            out[kw_norm] = {"volume": None, "kd": None, "cpc": None, "currency": "", "trend": ""}
    return out

def _safe_int(x):
    try: return int(float(x))
    except Exception: return None

def _safe_float(x):
    try: return float(x)
    except Exception: return None

def split_trend_string(trend_str: str) -> List[int]:
    if not trend_str: return []
    try: return [int(float(x)) for x in trend_str.split(",")]
    except Exception: return []
