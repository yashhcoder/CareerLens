"""
CareerLens - Career Skill-Gap & Job-Readiness Analyzer
=======================================================
Streamlit app for predicting career fit and surfacing skill gaps.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerLens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
}
.stApp { background: #f7f8fa; }

/* ── Header ── */
.cl-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563a8 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
}
.cl-header h1 { font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.cl-header p  { margin: 0.4rem 0 0; font-size: 1.05rem; opacity: 0.88; }

/* ── Section cards ── */
.cl-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.cl-card h3 { font-size: 1.05rem; font-weight: 700; color: #1f2328; margin: 0 0 0.8rem; }

/* ── Section labels ── */
.cl-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #57606a;
    margin-bottom: 0.3rem;
}

/* ── Result banner ── */
.cl-result-banner {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
    color: white;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1rem;
}
.cl-result-banner h2 { font-size: 1.6rem; font-weight: 800; margin: 0 0 0.2rem; }
.cl-result-banner .subtitle { font-size: 0.95rem; opacity: 0.85; }

/* ── Metric card ── */
.cl-metric {
    background: #f0f4ff;
    border: 1px solid #c7d7f5;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.cl-metric .val { font-size: 1.8rem; font-weight: 800; color: #2563a8; }
.cl-metric .lbl { font-size: 0.78rem; color: #57606a; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Skill pill (match) ── */
.skill-pill-match {
    display: inline-block;
    background: #d1fae5;
    border: 1px solid #6ee7b7;
    color: #065f46;
    border-radius: 99px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}
/* ── Skill pill (gap) ── */
.skill-pill-gap {
    display: inline-block;
    background: #fee2e2;
    border: 1px solid #fca5a5;
    color: #991b1b;
    border-radius: 99px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}
/* ── Priority card ── */
.priority-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
}
.priority-rank {
    background: #f59e0b;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 800; flex-shrink: 0;
}
.priority-name { font-weight: 600; color: #1f2328; font-size: 0.92rem; }
.priority-gap  { font-size: 0.78rem; color: #92400e; margin-top: 0.15rem; }

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563a8 0%, #3b82f6 100%);
    border-radius: 99px;
}

/* ── Top-match occupation rows ── */
.occ-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    border-bottom: 1px solid #f0f0f0;
    padding: 0.8rem 0;
}
.occ-rank {
    font-size: 1.4rem;
    font-weight: 800;
    color: #c9d1da;
    min-width: 2rem;
    text-align: right;
}
.occ-title { font-weight: 700; color: #1f2328; font-size: 0.95rem; }
.occ-meta  { font-size: 0.78rem; color: #57606a; margin-top: 0.15rem; }

/* ── Divider ── */
.cl-divider { height: 1px; background: #e5e7eb; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.pkl")

@st.cache_resource(show_spinner="Loading CareerLens model...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    artifact = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run `py train_model.py` first to train and save the model."
    )
    st.stop()

feature_cols      = artifact["feature_cols"]
feature_medians   = artifact["feature_medians"]
jz_pipeline       = artifact["jz_pipeline"]
jz_model_name     = artifact["jz_model_name"]
jz_model_results  = artifact["jz_model_results"]
knn_recommender   = artifact["knn_recommender"]
rec_scaler        = artifact["recommender_scaler"]
rec_X             = artifact["recommender_X"]
rec_titles        = artifact["recommender_titles"]
rec_codes         = artifact["recommender_codes"]
occ_profiles      = artifact["occ_profiles"]
all_skill_cols    = artifact["all_skill_cols"]
jz_reference      = artifact["jz_reference"]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

JZ_LABELS = {
    2: "Zone 2 — Some Preparation",
    3: "Zone 3 — Medium Preparation",
    4: "Zone 4 — Considerable Preparation",
    5: "Zone 5 — Extensive Preparation",
}
JZ_COLORS = {2: "#10b981", 3: "#3b82f6", 4: "#f59e0b", 5: "#ef4444"}
JZ_ICONS  = {2: "📗", 3: "📘", 4: "📙", 5: "📕"}


def build_user_vector(user_values: dict) -> pd.DataFrame:
    """Map user input dict → single-row DataFrame in the model's expected order."""
    row = {col: user_values.get(col, feature_medians.get(col, 0.0)) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def get_top_recommendations(user_vec: np.ndarray, k: int = 5):
    """Return top-k occupation matches using KNN cosine similarity."""
    scaled = rec_scaler.transform(user_vec)
    distances, indices = knn_recommender.kneighbors(scaled)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        title = rec_titles[idx]
        code  = rec_codes[idx]
        sim   = max(0.0, 1.0 - dist)  # cosine distance → similarity
        results.append({"title": title, "code": code, "similarity": sim})
    return results[:k]


def get_skill_gap(occ_title: str, user_values: dict):
    """
    Compare user's skill self-assessments against the occupation's required
    importance values. Returns (matched, gaps, priority_gaps).
    """
    row = occ_profiles[occ_profiles["Title"] == occ_title]
    if row.empty:
        return [], [], []

    row = row.iloc[0]
    match_skills = []
    gap_skills   = []

    for col in all_skill_cols:
        if col not in row:
            continue
        required = row[col]
        if pd.isna(required) or required < 2.5:   # skip low-importance skills
            continue
        skill_name = col.replace("Skill_", "")
        user_val   = user_values.get(col, feature_medians.get(col, 0.0))
        # Scale: dataset uses 1–5 importance; user input is 0–5
        if user_val >= required * 0.85:
            match_skills.append({"skill": skill_name, "required": required, "user": user_val})
        else:
            gap_skills.append({
                "skill": skill_name,
                "required": required,
                "user": user_val,
                "gap": required - user_val,
            })

    # Priority = biggest gap weighted by importance
    priority = sorted(gap_skills, key=lambda x: x["gap"] * x["required"], reverse=True)
    return match_skills, gap_skills, priority[:8]


def pretty_col(col: str) -> str:
    """Strip prefix and make column name human-readable."""
    for prefix in ("Skill_", "Ability_", "Knowledge_", "Style_", "Interest_"):
        if col.startswith(prefix):
            return col[len(prefix):]
    return col


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cl-header">
    <h1>&#128301; CareerLens</h1>
    <p>Enter your skill levels below and CareerLens will predict your career readiness zone,
    recommend the most suitable occupations, and surface your personal skill gaps.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL INFO RIBBON
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Model Info", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Classifier", jz_model_name)
    with col2:
        best_acc = jz_model_results[jz_model_name]["test_accuracy"]
        st.metric("Test Accuracy", f"{best_acc:.1%}")
    with col3:
        best_cv = jz_model_results[jz_model_name]["cv_mean"]
        st.metric("CV Accuracy", f"{best_cv:.1%}")

    st.caption("Model Comparison")
    rows = []
    for name, res in jz_model_results.items():
        rows.append({
            "Model": name,
            "Test Accuracy": f"{res['test_accuracy']:.1%}",
            "CV Mean": f"{res['cv_mean']:.1%}",
            "CV Std": f"±{res['cv_std']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Your Skill Profile")
    st.caption("Rate each area 0 (none) to 5 (expert). Adjust sliders to match your experience.")
    st.markdown("---")

# Group feature columns by category for display
interest_cols  = [c for c in feature_cols if c.startswith("Interest_")]
skill_cols     = [c for c in feature_cols if c.startswith("Skill_")]
ability_cols   = [c for c in feature_cols if c.startswith("Ability_")]
knowledge_cols = [c for c in feature_cols if c.startswith("Knowledge_")]
style_cols     = [c for c in feature_cols if c.startswith("Style_")]

# ──────────────────────────────────────────────────────────────────────────────
# INPUT TABS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="cl-section-label">Step 1 — Rate your skills across five categories</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Career Interests", "📚 Essential Skills", "🧠 Abilities", "💡 Knowledge", "🌟 Work Styles"
])

user_values = {}

# ── Tab 1: RIASEC Interests ──
with tab1:
    st.markdown("**RIASEC Interest Profile** — Rate your level of interest in each area (0 = not at all, 7 = very strong)")
    cols = st.columns(2)
    RIASEC_DESC = {
        "Realistic":     "Working with tools, machines, nature",
        "Investigative": "Research, analysis, problem-solving",
        "Artistic":      "Creative expression, design, arts",
        "Social":        "Helping, teaching, working with people",
        "Enterprising":  "Leading, persuading, business",
        "Conventional":  "Organizing, data, structured tasks",
    }
    for i, col in enumerate(interest_cols):
        name = pretty_col(col)
        desc = RIASEC_DESC.get(name, "")
        default_val = float(feature_medians.get(col, 3.5))
        with cols[i % 2]:
            val = st.slider(
                f"{name}",
                min_value=0.0, max_value=7.0, value=default_val, step=0.1,
                help=desc, key=f"input_{col}"
            )
            user_values[col] = val

# ── Tab 2: Essential Skills ──
with tab2:
    st.markdown("**Essential Skills** — Rate your proficiency (0 = no skill, 5 = expert)")
    cols = st.columns(2)
    for i, col in enumerate(skill_cols):
        name = pretty_col(col)
        default_val = float(feature_medians.get(col, 2.5))
        with cols[i % 2]:
            val = st.slider(name, 0.0, 5.0, default_val, 0.1, key=f"input_{col}")
            user_values[col] = val

# ── Tab 3: Abilities ──
with tab3:
    st.markdown("**Cognitive & Physical Abilities** — Rate your proficiency (0 = low, 5 = high)")
    cols = st.columns(2)
    for i, col in enumerate(ability_cols):
        name = pretty_col(col)
        default_val = float(feature_medians.get(col, 2.5))
        with cols[i % 2]:
            val = st.slider(name, 0.0, 5.0, default_val, 0.1, key=f"input_{col}")
            user_values[col] = val

# ── Tab 4: Knowledge ──
with tab4:
    st.markdown("**Knowledge Areas** — How strong is your knowledge in each domain? (0 = none, 5 = expert)")
    cols = st.columns(2)
    for i, col in enumerate(knowledge_cols):
        name = pretty_col(col)
        default_val = float(feature_medians.get(col, 2.5))
        with cols[i % 2]:
            val = st.slider(name, 0.0, 5.0, default_val, 0.1, key=f"input_{col}")
            user_values[col] = val

# ── Tab 5: Work Styles ──
with tab5:
    st.markdown("**Work Style Preferences** — How strongly do these work traits describe you? (0 = not at all, 3 = very much)")
    cols = st.columns(2)
    for i, col in enumerate(style_cols):
        name = pretty_col(col)
        default_val = float(feature_medians.get(col, 1.5))
        with cols[i % 2]:
            val = st.slider(name, 0.0, 3.0, min(default_val, 3.0), 0.05, key=f"input_{col}")
            user_values[col] = val


# ──────────────────────────────────────────────────────────────────────────────
# ANALYZE BUTTON
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='cl-divider'></div>", unsafe_allow_html=True)
st.markdown('<p class="cl-section-label">Step 2 — Run the analysis</p>', unsafe_allow_html=True)

analyze_clicked = st.button(
    "🔍  Analyze My Career",
    type="primary",
    use_container_width=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────────────────────────────
if analyze_clicked:
    user_vec = build_user_vector(user_values)

    # ── Job Zone Prediction ──
    jz_pred = int(jz_pipeline.predict(user_vec)[0])
    jz_proba = jz_pipeline.predict_proba(user_vec)[0]
    jz_classes = jz_pipeline.classes_
    jz_confidence = float(jz_proba.max())

    # ── Top Occupations ──
    top_matches = get_top_recommendations(user_vec, k=5)
    top_occ_title = top_matches[0]["title"] if top_matches else None

    # ── Skill Gap for top match ──
    matched_skills, gap_skills, priority_gaps = [], [], []
    if top_occ_title:
        matched_skills, gap_skills, priority_gaps = get_skill_gap(top_occ_title, user_values)

    # ── RESULT BANNER ──
    jz_icon  = JZ_ICONS.get(jz_pred, "🔵")
    jz_label = JZ_LABELS.get(jz_pred, f"Zone {jz_pred}")
    st.markdown(f"""
    <div class="cl-result-banner">
        <h2>{jz_icon} Career Readiness: {jz_label}</h2>
        <div class="subtitle">
            Based on your skill profile, you match occupations in <strong>Job Zone {jz_pred}</strong>.
            Model confidence: <strong>{jz_confidence:.0%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary Metrics ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="cl-metric"><div class="val">{jz_pred}</div>
        <div class="lbl">Job Zone</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="cl-metric"><div class="val">{jz_confidence:.0%}</div>
        <div class="lbl">Confidence</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="cl-metric"><div class="val">{len(matched_skills)}</div>
        <div class="lbl">Skills Matched</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="cl-metric"><div class="val">{len(gap_skills)}</div>
        <div class="lbl">Skill Gaps</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='cl-divider'></div>", unsafe_allow_html=True)

    # ── Zone Probability Bar ──
    st.markdown("#### Job Zone Probability Distribution")
    zone_prob_data = {f"Zone {cls}": float(prob) for cls, prob in zip(jz_classes, jz_proba)}
    prob_df = pd.DataFrame(list(zone_prob_data.items()), columns=["Zone", "Probability"])
    prob_df = prob_df.sort_values("Zone")
    st.bar_chart(prob_df.set_index("Zone")["Probability"], use_container_width=True, height=200)

    # ── Zone Description ──
    jz_desc = {
        2: "Occupations here require little to some preparation. Typically a high school diploma and short on-the-job training.",
        3: "Medium preparation needed. Usually requires vocational training or an associate's degree and some related experience.",
        4: "Considerable preparation needed. Most require a bachelor's degree and several years of work-related experience.",
        5: "Extensive preparation needed. Usually requires graduate school (master's, PhD, MD, etc.) and extensive experience.",
    }
    if jz_pred in jz_desc:
        st.info(f"**About Job Zone {jz_pred}:** {jz_desc[jz_pred]}")

    st.markdown("<div class='cl-divider'></div>", unsafe_allow_html=True)

    # ── Top Occupation Matches ──
    st.markdown("#### 🎯 Top Occupation Matches")
    st.caption("Ranked by cosine similarity between your skill profile and each occupation's O\\*NET profile.")

    for rank, occ in enumerate(top_matches, 1):
        occ_row = occ_profiles[occ_profiles["Title"] == occ["title"]]
        jz_for_occ = int(occ_row["Job Zone"].values[0]) if not occ_row.empty else "?"
        desc_for_occ = occ_row["Description"].values[0][:160] + "..." if not occ_row.empty else ""
        badge_color = JZ_COLORS.get(jz_for_occ, "#888")
        sim_pct = f"{occ['similarity']:.0%}"

        st.markdown(f"""
        <div class="occ-row">
            <div class="occ-rank">#{rank}</div>
            <div style="flex:1">
                <div class="occ-title">{occ['title']}</div>
                <div class="occ-meta">{desc_for_occ}</div>
                <div style="margin-top:0.4rem">
                    <span style="background:{badge_color};color:white;border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:700;">
                        Zone {jz_for_occ}
                    </span>
                    &nbsp;
                    <span style="background:#e0eeff;color:#1d4ed8;border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:700;">
                        {sim_pct} match
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='cl-divider'></div>", unsafe_allow_html=True)

    # ── Skill-Gap Analysis (focused on top match) ──
    if top_occ_title:
        st.markdown(f"#### 🏆 Skill Analysis for: *{top_occ_title}*")
        st.caption("Comparing your self-rated skills against this occupation's O\\*NET importance scores.")

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("**✅ Skills You Already Match**")
            if matched_skills:
                pills_html = "".join(
                    f'<span class="skill-pill-match">{s["skill"]}</span>'
                    for s in matched_skills
                )
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.caption("No strong skill matches found for this occupation. Try adjusting your ratings.")

        with right_col:
            st.markdown("**⚠️ Skills You Are Missing or Need to Improve**")
            if gap_skills:
                pills_html = "".join(
                    f'<span class="skill-pill-gap">{s["skill"]}</span>'
                    for s in gap_skills
                )
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.caption("Great — your skills strongly match this occupation's requirements!")

        st.markdown("<div class='cl-divider'></div>", unsafe_allow_html=True)

        # ── What Should I Learn Next ──
        st.markdown("#### 🚀 What Should I Learn Next?")
        st.caption(
            "Prioritized by gap size × importance — highest-impact skills to develop for your top career match."
        )

        if priority_gaps:
            for i, item in enumerate(priority_gaps, 1):
                gap_pct = min(1.0, item["gap"] / 5.0)
                user_pct = min(1.0, item["user"] / 5.0)
                req_pct  = min(1.0, item["required"] / 5.0)
                st.markdown(f"""
                <div class="priority-item">
                    <div class="priority-rank">{i}</div>
                    <div style="flex:1">
                        <div class="priority-name">{item["skill"]}</div>
                        <div class="priority-gap">
                            Your level: {item["user"]:.1f} / 5 &nbsp;|&nbsp;
                            Required: {item["required"]:.1f} / 5 &nbsp;|&nbsp;
                            Gap: {item["gap"]:.1f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(user_pct, text=f"Your level: {user_pct:.0%} of required")
        else:
            st.success("Your skills already meet or exceed the requirements for this occupation!")

    st.markdown("<div class='cl-divider'></div>", unsafe_allow_html=True)

    # ── Full Skills Comparison Table ──
    if top_occ_title and (matched_skills or gap_skills):
        with st.expander("Full Skills Detail Table", expanded=False):
            all_skills = matched_skills + gap_skills
            skill_df = pd.DataFrame(all_skills)
            if not skill_df.empty:
                skill_df = skill_df.rename(columns={
                    "skill": "Skill",
                    "required": "Required (O*NET)",
                    "user": "Your Rating",
                    "gap": "Gap",
                })
                skill_df = skill_df.sort_values("Required (O*NET)", ascending=False).reset_index(drop=True)
                if "Gap" not in skill_df.columns:
                    skill_df["Gap"] = skill_df["Required (O*NET)"] - skill_df["Your Rating"]
                st.dataframe(
                    skill_df.style.format({
                        "Required (O*NET)": "{:.2f}",
                        "Your Rating": "{:.1f}",
                        "Gap": "{:.2f}",
                    }).background_gradient(subset=["Gap"], cmap="RdYlGn_r"),
                    use_container_width=True,
                )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
            text-align: center; font-size: 0.78rem; color: #57606a;">
    CareerLens &nbsp;|&nbsp; Powered by O*NET occupational data &nbsp;|&nbsp;
    Random Forest classifier &nbsp;+&nbsp; KNN occupation recommender
</div>
""", unsafe_allow_html=True)
