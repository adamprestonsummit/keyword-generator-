import os, json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from utils_scrape import extract_text_from_url
from utils_llm import llm_expand_keywords, llm_tag_batch
from utils_text import normalize_kw, dedupe_keywords
from utils_semrush import get_semrush_metrics, db_for_market, split_trend_string

st.set_page_config(page_title="Keyword Generator", page_icon="🔑", layout="wide")
st.title("🔑 SEO Keyword Generator")
st.caption("Seed prompt or URL → expand → dedupe → cluster → auto-tag → optional Semrush volumes → export")

# --- Secrets / keys ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
SEMRUSH_API_KEY = st.secrets.get("SEMRUSH_API_KEY") or os.getenv("SEMRUSH_API_KEY")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Generation")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Gemini"])
    temperature = st.slider("Creativity (temp)", 0.0, 1.0, 0.2, 0.1)
    n_ideas = st.slider("Idea breadth", 30, 300, 120, 10)
    audience = st.text_input("Audience (optional)", "UK consumers")
    market = st.text_input("Market/Locale", "United Kingdom")
    language = st.text_input("Language", "English")
    dedupe_thresh = st.slider("Dedupe cosine threshold", 0.80, 0.99, 0.90, 0.01)
    cluster_method = st.selectbox("Clustering", ["HDBSCAN (auto)", "KMeans (auto-k)"])
    st.markdown("---")
    st.subheader("Semrush (optional)")
    use_semrush = st.checkbox("Attach Semrush volumes (Volume, KD, CPC, Trend)", value=False)
    max_semrush = st.slider("Max keywords to enrich", 50, 1000, 200, 50, disabled=not use_semrush)
    include_trend = st.checkbox("Include 12-month Trend", value=True, disabled=not use_semrush)
    st.caption(f"Database autodetected from Market → '{db_for_market(market)}'. Override in code if needed.")
    st.markdown("---")
    st.caption("Tip: keep temp low for consistent tagging & clustering.")

tab1, tab2 = st.tabs(["Seed prompt", "URL/domain"])

seed_prompt = None
url_input = None

with tab1:
    seed_prompt = st.text_area("Seed prompt", placeholder="e.g., winter gardening tips for tradespeople, DIY focus")
    gen_btn1 = st.button("Generate from seed prompt", type="primary")

with tab2:
    url_input = st.text_input("URL or domain", placeholder="https://www.wickes.co.uk")
    gen_btn2 = st.button("Generate from URL", type="primary")

def ensure_llm_keys():
    if provider == "OpenAI" and not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY."); return False
    if provider == "Gemini" and not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY."); return False
    return True

triggered = (gen_btn1 and seed_prompt) or (gen_btn2 and url_input)
if triggered:
    if not ensure_llm_keys():
        st.stop()

    # 1) Collect source context (if URL provided)
    with st.spinner("Collecting source context…"):
        source_text = extract_text_from_url(url_input) if url_input else ""

    # 2) Generate keyword ideas
    with st.spinner("Generating keyword ideas…"):
        raw_ideas = llm_expand_keywords(
            provider=provider,
            api_key=OPENAI_API_KEY if provider=="OpenAI" else GEMINI_API_KEY,
            n_ideas=n_ideas,
            temperature=temperature,
            seed=seed_prompt,
            source_text=source_text,
            audience=audience,
            market=market,
            language=language
        )
        keywords = [normalize_kw(k) for k in raw_ideas if k]

    # 3) Dedupe (embeddings)
    embs = embedder.encode(keywords, normalize_embeddings=True, show_progress_bar=False)
    keep_mask = dedupe_keywords(keywords, embs, threshold=dedupe_thresh)
    kw_deduped = [k for k,m in zip(keywords, keep_mask) if m]
    embs_dedup = embs[keep_mask]

    # 4) Cluster
    with st.spinner("Clustering…"):
        labels = np.array([-1]*len(kw_deduped))
        if cluster_method.startswith("HDBSCAN"):
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
                labels = clusterer.fit_predict(embs_dedup)
            except Exception:
                pass
        else:
            # Auto-k via silhouette
            best_k, best_score, best_labels = None, -1, None
            for k in range(4, min(25, max(5, len(kw_deduped)//8))):
                mdl = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(embs_dedup)
                score = silhouette_score(embs_dedup, mdl.labels_) if len(set(mdl.labels_))>1 else -1
                if score > best_score:
                    best_k, best_score, best_labels = k, score, mdl.labels_
            if best_labels is not None:
                labels = best_labels

    # 5) Auto-tag (heuristics + LLM refinement)
    with st.spinner("Auto-tagging…"):
        df = pd.DataFrame({"keyword": kw_deduped, "cluster": labels})

        def quick_intent(k):
            t = k.lower()
            if any(w in t for w in ["buy","price","deal","near me"]): return "Transactional"
            if any(w in t for w in ["best","top","vs","compare","review"]): return "Commercial"
            if any(w in t for w in ["how","what","why","guide","ideas","tips"]): return "Informational"
            if any(w in t for w in ["login","contact","brand","homepage"]): return "Navigational"
            return "Informational"
        df["intent_rule"] = df["keyword"].map(quick_intent)

        tags = llm_tag_batch(
            provider=provider,
            api_key=OPENAI_API_KEY if provider=="OpenAI" else GEMINI_API_KEY,
            keywords=df["keyword"].tolist(),
            market=market,
            language=language
        )
      
    tag_df = pd.DataFrame(tags)

    # Make sure expected columns exist even if the LLM returned weird data:
    for col in ["keyword", "intent", "funnel", "theme"]:
        if col not in tag_df.columns:
            tag_df[col] = np.nan

    # Keep only the columns we care about, one row per keyword
    tag_df = tag_df[["keyword", "intent", "funnel", "theme"]].drop_duplicates(subset=["keyword"])

    # Merge on 'keyword' to avoid overlapping-column error
    df = df.merge(tag_df, on="keyword", how="left")

    # 6) Name clusters
    cluster_names = {}
    for c in sorted(df["cluster"].unique()):
        kws = df.loc[df["cluster"]==c, "keyword"].tolist()
        if c == -1 or not kws:
            cluster_names[c] = "Misc / Unclustered"
        else:
            terms = pd.Series(" ".join(kws).split()).value_counts().head(3).index.tolist()
            cluster_names[c] = ", ".join(terms)
    df["topic"] = df["cluster"].map(cluster_names)

    # 7) Optional: Semrush enrichment
    semrush_units = 0
    if use_semrush:
        if not SEMRUSH_API_KEY:
            st.warning("Semrush is enabled but no SEMRUSH_API_KEY found. Skipping enrichment.")
        else:
            subset = df["keyword"].head(max_semrush).tolist()
            st.info(f"Fetching Semrush metrics for {len(subset)} keywords in '{db_for_market(market)}'…")
            metrics = get_semrush_metrics(
                api_key=SEMRUSH_API_KEY,
                keywords=subset,
                market=market,
                per_call_sleep=0.2,
                with_trend=include_trend
            )
            sem_df = (
                pd.DataFrame.from_dict(metrics, orient="index")
                .reset_index()
                .rename(columns={"index": "keyword"})
            )
            df = df.merge(sem_df, on="keyword", how="left")
            # rough unit counter: phrase_all + phrase_kdi + (optional) trends
            semrush_units = len(subset) * (2 + (1 if include_trend else 0))

            # Optional: split trend into columns
            if include_trend and "trend" in df.columns:
                trend_cols = [f"trend_m{i+1}" for i in range(12)]
                trends = df["trend"].apply(split_trend_string)
                for i in range(12):
                    df[trend_cols[i]] = trends.apply(lambda arr: arr[i] if len(arr)>i else None)

    # 8) Output
    st.success(f"Generated {len(df)} unique keywords across {df['cluster'].nunique()} clusters.")
    if use_semrush:
        st.caption(f"Semrush units (approx): {semrush_units}")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="keywords.csv", mime="text/csv")
