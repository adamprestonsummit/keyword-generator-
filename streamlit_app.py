import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from utils_scrape import extract_text_from_url
from utils_llm import llm_expand_keywords, llm_tag_batch
from utils_text import normalize_kw, dedupe_keywords
from utils_semrush import get_semrush_metrics, db_for_market, split_trend_string

# ---------------------------
# App setup
# ---------------------------
st.set_page_config(page_title="Keyword Generator", page_icon="🔑", layout="wide")
st.title("🔑 SEO Keyword Generator")
st.caption("Seed prompt or URL(s) → expand → dedupe → cluster → auto-tag → optional Semrush volumes → export")

# Secrets / keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
SEMRUSH_API_KEY = st.secrets.get("SEMRUSH_API_KEY") or os.getenv("SEMRUSH_API_KEY")

# Lazy-load embedder (faster first boot)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Cache page extraction per URL (saves time/requests)
@st.cache_data(show_spinner=False)
def cached_extract_text(url: str) -> str:
    return extract_text_from_url(url)

def ensure_llm_keys(provider: str) -> bool:
    if provider == "OpenAI" and not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY.")
        return False
    if provider == "Gemini" and not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY.")
        return False
    return True

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.subheader("Generation")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini"])
    temperature = st.slider("Creativity (temp)", 0.0, 1.0, 0.2, 0.1)
    n_ideas = st.slider("Idea breadth", 30, 300, 120, 10)
    audience = st.text_input("Audience (optional)", "UK consumers")
    market = st.text_input("Market/Locale", "United Kingdom")
    language = st.text_input("Language", "English")
    dedupe_thresh = st.slider("Dedupe cosine threshold", 0.80, 0.99, 0.90, 0.01)
    cluster_method = st.selectbox("Clustering", ["KMeans (auto-k)", "HDBSCAN (auto)"])
    st.markdown("---")
    st.subheader("Semrush (optional)")
    use_semrush = st.checkbox("Attach Semrush volumes (Volume, KD, CPC, Trend)", value=False)
    max_semrush = st.slider("Max keywords to enrich", 50, 1000, 200, 50, disabled=not use_semrush)
    include_trend = st.checkbox("Include 12-month Trend", value=True, disabled=not use_semrush)
    st.caption(f"Database autodetected from Market → '{db_for_market(market)}'.")
    st.markdown("---")
    st.caption("Tip: keep temp low for consistent tagging & clustering.")

# ---------------------------
# Inputs (tabs)
# ---------------------------
tab1, tab2 = st.tabs(["Seed prompt", "URL(s)"])

with tab1:
    seed_prompt = st.text_area(
        "Seed prompt",
        placeholder="e.g., winter gardening tips for tradespeople, DIY focus"
    )
    gen_btn_seed = st.button("Generate from seed prompt", type="primary")

with tab2:
    url_input = st.text_area(
        "One or more URLs (one per line)",
        placeholder="https://www.example.com/page-1\nhttps://www.example.com/page-2"
    )
    gen_btn_urls = st.button("Generate from URL(s)", type="primary")

# ---------------------------
# Core processing for a single source (seed or one URL)
# ---------------------------
def process_source(source_label: str, source_text: str, seed_prompt_val: str | None) -> pd.DataFrame:
    """
    Pipeline for one source: expand -> dedupe -> cluster -> tag.
    Returns DataFrame with columns: keyword, cluster, source_url, intent_rule, intent, funnel, theme, topic
    """
    # 1) Generate ideas
    raw_ideas = llm_expand_keywords(
        provider=provider,
        api_key=OPENAI_API_KEY if provider == "OpenAI" else GEMINI_API_KEY,
        n_ideas=n_ideas,
        temperature=temperature,
        seed=seed_prompt_val,
        source_text=source_text,
        audience=audience,
        market=market,
        language=language,
    )
    keywords = [normalize_kw(k) for k in raw_ideas if k]

    # 2) Dedupe
    embedder = load_embedder()
    embs = embedder.encode(keywords, normalize_embeddings=True, show_progress_bar=False)
    keep_mask = dedupe_keywords(keywords, embs, threshold=dedupe_thresh)
    kw_deduped = [k for k, m in zip(keywords, keep_mask) if m]
    embs_dedup = embs[keep_mask]

    # 3) Cluster
    labels = np.array([-1] * len(kw_deduped))
    if cluster_method.startswith("HDBSCAN"):
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
            labels = clusterer.fit_predict(embs_dedup)
        except Exception:
            pass
    else:
        # Auto-k via silhouette across a small range
        best_k, best_score, best_labels = None, -1, None
        k_max = min(25, max(5, len(kw_deduped) // 8))
        for k in range(4, k_max):
            mdl = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(embs_dedup)
            score = silhouette_score(embs_dedup, mdl.labels_) if len(set(mdl.labels_)) > 1 else -1
            if score > best_score:
                best_k, best_score, best_labels = k, score, mdl.labels_
        if best_labels is not None:
            labels = best_labels

    # 4) Heuristic intent + LLM tags
    df = pd.DataFrame({"keyword": kw_deduped, "cluster": labels, "source_url": source_label})

    def quick_intent(k):
        t = k.lower()
        if any(w in t for w in ["buy", "price", "deal", "near me"]):
            return "Transactional"
        if any(w in t for w in ["best", "top", "vs", "compare", "review"]):
            return "Commercial"
        if any(w in t for w in ["how", "what", "why", "guide", "ideas", "tips"]):
            return "Informational"
        if any(w in t for w in ["login", "contact", "brand", "homepage"]):
            return "Navigational"
        return "Informational"

    df["intent_rule"] = df["keyword"].map(quick_intent)

    tags = llm_tag_batch(
        provider=provider,
        api_key=OPENAI_API_KEY if provider == "OpenAI" else GEMINI_API_KEY,
        keywords=df["keyword"].tolist(),
        market=market,
        language=language,
    )

    # SAFE merge to avoid "columns overlap" error
    tag_df = pd.DataFrame(tags)
    for col in ["keyword", "intent", "funnel", "theme"]:
        if col not in tag_df.columns:
            tag_df[col] = np.nan
    tag_df = tag_df[["keyword", "intent", "funnel", "theme"]].drop_duplicates(subset=["keyword"])
    df = df.merge(tag_df, on="keyword", how="left")

    # 5) Name clusters (per source)
    cluster_names = {}
    for c in sorted(df["cluster"].unique()):
        kws = df.loc[df["cluster"] == c, "keyword"].tolist()
        if c == -1 or not kws:
            cluster_names[c] = "Misc / Unclustered"
        else:
            terms = pd.Series(" ".join(kws).split()).value_counts().head(3).index.tolist()
            cluster_names[c] = ", ".join(terms)
    df["topic"] = df["cluster"].map(cluster_names)
    return df

# ---------------------------
# Trigger & pipeline
# ---------------------------
triggered = (gen_btn_seed and seed_prompt) or (gen_btn_urls and url_input)
if triggered:
    if not ensure_llm_keys(provider):
        st.stop()

    all_rows = []

    # Seed path
    if gen_btn_seed and seed_prompt:
        with st.spinner("Generating from seed prompt…"):
            df_seed = process_source(
                source_label=f"SEED: {seed_prompt[:50]}…",
                source_text="",
                seed_prompt_val=seed_prompt,
            )
        all_rows.append(df_seed)

    # Multi-URL path
    if gen_btn_urls and url_input:
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        with st.spinner(f"Collecting and generating from {len(urls)} URL(s)…"):
            for u in urls:
                page_text = cached_extract_text(u)
                df_url = process_source(source_label=u, source_text=page_text, seed_prompt_val=None)
                all_rows.append(df_url)

    if not all_rows:
        st.warning("No inputs detected.")
        st.stop()

    # Combine results from all sources
    df = pd.concat(all_rows, ignore_index=True)

    # Optional: Semrush enrichment on unique keywords (saves units)
    semrush_units = 0
    if use_semrush:
        if not SEMRUSH_API_KEY:
            st.warning("Semrush is enabled but no SEMRUSH_API_KEY found. Skipping enrichment.")
        else:
            uniq = df["keyword"].drop_duplicates().head(max_semrush).tolist()
            st.info(f"Fetching Semrush metrics for {len(uniq)} unique keywords in '{db_for_market(market)}'…")
            metrics = get_semrush_metrics(
                api_key=SEMRUSH_API_KEY,
                keywords=uniq,
                market=market,
                per_call_sleep=0.2,
                with_trend=include_trend,
            )
            sem_df = (
                pd.DataFrame.from_dict(metrics, orient="index")
                .reset_index()
                .rename(columns={"index": "keyword"})
            )
            df = df.merge(sem_df, on="keyword", how="left")
            semrush_units = len(uniq) * (2 + (1 if include_trend else 0))

            # Split trend into 12 columns (optional)
            if include_trend and "trend" in df.columns:
                trend_cols = [f"trend_m{i+1}" for i in range(12)]
                trends = df["trend"].apply(split_trend_string)
                for i in range(12):
                    df[trend_cols[i]] = trends.apply(lambda arr: arr[i] if len(arr) > i else None)

    # Output & filters
    st.success(
        f"Generated {len(df)} rows across {df['cluster'].nunique()} clusters "
        f"from {df['source_url'].nunique()} source(s)."
    )
    if use_semrush:
        st.caption(f"Semrush units (approx): {semrush_units}")

    with st.expander("Filters"):
        src_pick = st.multiselect(
            "Filter by source URL",
            options=sorted(df["source_url"].unique()),
            default=list(sorted(df["source_url"].unique())),
        )
        df_view = df[df["source_url"].isin(src_pick)]

    st.dataframe(df_view, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="keywords_by_url.csv", mime="text/csv")
