"""
CareerLens - ML Training Pipeline
===================================
Trains two ML components:
  1. Job Zone Classifier (classification, Job Zone 1-5 target)
     → Compared: Random Forest, Gradient Boosting, KNN
  2. Occupation Recommender (KNN similarity index stored alongside the classifier)
     → Finds top-N most similar occupations to a user's skill profile

Features: RIASEC interest scores + top essential skills (IM) +
          top abilities (IM) + top knowledge (IM) + work style impact (WI)

Saves artifacts to models/model.pkl
"""

import os
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def pivot_long_to_wide(filepath: str, scale_id: str, prefix: str,
                       value_col: str = "Data Value",
                       element_col: str = "Element Name",
                       occ_col: str = "O*NET-SOC Code") -> pd.DataFrame:
    """
    Reads a long-format O*NET CSV, filters to one Scale ID,
    pivots Element Name → wide columns.
    """
    df = pd.read_csv(filepath)
    df = df[df["Scale ID"] == scale_id].copy()
    # Filter out suppressed records if the column exists
    if "Recommend Suppress" in df.columns:
        df = df[df["Recommend Suppress"] != "Y"]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    df = df[[occ_col, element_col, value_col]].drop_duplicates(
        subset=[occ_col, element_col], keep="first"
    )
    pivot = df.pivot(index=occ_col, columns=element_col, values=value_col)
    pivot.columns = [f"{prefix}_{c}" for c in pivot.columns]
    return pivot.reset_index().rename(columns={occ_col: "O*NET-SOC Code"})


def load_riasec() -> pd.DataFrame:
    """Load RIASEC interest scores for each occupation."""
    filepath = os.path.join(DATA_DIR, "career_interest_types.csv")
    df = pd.read_csv(filepath)
    riasec = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]
    df = df[(df["Scale ID"] == "OI") & (df["Element Name"].isin(riasec))].copy()
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df = df.dropna(subset=["Data Value"])
    df = df[["O*NET-SOC Code", "Element Name", "Data Value"]].drop_duplicates(
        subset=["O*NET-SOC Code", "Element Name"], keep="first"
    )
    pivot = df.pivot(index="O*NET-SOC Code", columns="Element Name", values="Data Value")
    pivot.columns = [f"Interest_{c}" for c in pivot.columns]
    return pivot.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# BUILD FEATURE MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix() -> tuple:
    print("Loading occupation data...")
    occ_df = pd.read_csv(os.path.join(DATA_DIR, "occupation_data.csv"))
    occ_df = occ_df[["O*NET-SOC Code", "Title", "Description"]].copy()
    print(f"  {len(occ_df)} occupations")

    print("Loading job zones...")
    jz_df = pd.read_csv(os.path.join(DATA_DIR, "job_zones.csv"))
    jz_df = jz_df[["O*NET-SOC Code", "Job Zone"]].drop_duplicates(subset=["O*NET-SOC Code"])
    jz_df["Job Zone"] = pd.to_numeric(jz_df["Job Zone"], errors="coerce")
    jz_df = jz_df.dropna(subset=["Job Zone"])
    jz_df["Job Zone"] = jz_df["Job Zone"].astype(int)
    print(f"  {len(jz_df)} job zone records")

    print("Loading RIASEC interests...")
    riasec_df = load_riasec()
    print(f"  {len(riasec_df)} RIASEC records")

    print("Loading essential skills (Importance)...")
    skills_df = pivot_long_to_wide(
        os.path.join(DATA_DIR, "essential_skills.csv"), "IM", "Skill"
    )
    print(f"  {len(skills_df)} occupation-skill records, {len(skills_df.columns)-1} skills")

    print("Loading abilities (Importance)...")
    abilities_df = pivot_long_to_wide(
        os.path.join(DATA_DIR, "abilities.csv"), "IM", "Ability"
    )
    print(f"  {len(abilities_df)} occupation-ability records, {len(abilities_df.columns)-1} abilities")

    print("Loading knowledge (Importance)...")
    knowledge_df = pivot_long_to_wide(
        os.path.join(DATA_DIR, "knowledge.csv"), "IM", "Knowledge"
    )
    print(f"  {len(knowledge_df)} occupation-knowledge records, {len(knowledge_df.columns)-1} knowledge areas")

    print("Loading work styles (Work Styles Impact)...")
    styles_df = pivot_long_to_wide(
        os.path.join(DATA_DIR, "work_styles.csv"), "WI", "Style"
    )
    print(f"  {len(styles_df)} occupation-style records, {len(styles_df.columns)-1} styles")

    # ── JOIN ──
    print("\nJoining tables...")
    merged = occ_df.copy()
    for tbl, name in [
        (jz_df, "Job Zones"),
        (riasec_df, "RIASEC"),
        (skills_df, "Skills"),
        (abilities_df, "Abilities"),
        (knowledge_df, "Knowledge"),
        (styles_df, "Work Styles"),
    ]:
        before = len(merged)
        merged = merged.merge(tbl, on="O*NET-SOC Code", how="inner")
        print(f"  After {name}: {len(merged)} rows (was {before})")

    print(f"\nFinal dataset: {len(merged)} occupations, {len(merged.columns)} columns")

    # ── SELECT FEATURES ──
    # 6 RIASEC + up to 10 skills + up to 10 abilities + up to 10 knowledge + up to 8 styles
    def top_n_cols(df_merged, prefix, n):
        cols = [c for c in df_merged.columns if c.startswith(prefix + "_")]
        if not cols:
            return []
        variance = df_merged[cols].var()
        return variance.nlargest(n).index.tolist()

    feature_cols = []
    interest_cols = [c for c in merged.columns if c.startswith("Interest_")]
    feature_cols.extend(interest_cols)
    feature_cols.extend(top_n_cols(merged, "Skill", 10))
    feature_cols.extend(top_n_cols(merged, "Ability", 10))
    feature_cols.extend(top_n_cols(merged, "Knowledge", 10))
    feature_cols.extend(top_n_cols(merged, "Style", 8))

    # Remove duplicates
    seen = set()
    unique_features = []
    for f in feature_cols:
        if f not in seen and f in merged.columns:
            seen.add(f)
            unique_features.append(f)

    print(f"Selected {len(unique_features)} features for ML.")

    # All skill cols for skill-gap analysis
    all_skill_cols = [c for c in merged.columns if c.startswith("Skill_")]
    all_ability_cols = [c for c in merged.columns if c.startswith("Ability_")]
    all_knowledge_cols = [c for c in merged.columns if c.startswith("Knowledge_")]

    return merged, unique_features, all_skill_cols, all_ability_cols, all_knowledge_cols


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN & COMPARE MODELS
# ──────────────────────────────────────────────────────────────────────────────

def train_job_zone_classifier(X_train, X_test, y_train, y_test):
    """Train multiple classifiers for Job Zone prediction, return best pipeline."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, metric="euclidean"
        ),
    }

    print("\n" + "=" * 60)
    print("JOB ZONE CLASSIFIER - MODEL COMPARISON")
    print("=" * 60)

    results = {}
    best_name, best_acc, best_pipeline = None, -1, None

    for name, model in models.items():
        print(f"\n>> Training {name}...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        results[name] = {"test_accuracy": acc, "cv_mean": cv.mean(), "cv_std": cv.std()}
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  CV  Accuracy  : {cv.mean():.4f} ± {cv.std():.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_name} (test accuracy = {best_acc:.4f})")
    return best_pipeline, best_name, results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CareerLens - ML Training Pipeline")
    print("=" * 60)

    merged, feature_cols, all_skill_cols, all_ability_cols, all_knowledge_cols = build_feature_matrix()

    X_all = merged[feature_cols].copy().fillna(merged[feature_cols].median(numeric_only=True))
    y_jz = merged["Job Zone"].values

    # Job Zone stratified split (5 classes with many samples each → valid)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_jz, test_size=0.2, random_state=42, stratify=y_jz
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Job Zone distribution: {pd.Series(y_jz).value_counts().sort_index().to_dict()}")

    # Train & select best Job Zone classifier
    best_pipeline, best_name, results = train_job_zone_classifier(X_train, X_test, y_train, y_test)

    # ── KNN OCCUPATION RECOMMENDER ──
    print("\nBuilding occupation recommender (KNN similarity)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    n_neighbors = min(10, len(merged))
    knn_recommender = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn_recommender.fit(X_scaled)
    print(f"  KNN recommender fitted on {len(X_all)} occupation profiles, k={n_neighbors}")

    # ── FEATURE MEDIANS (for imputation at inference) ──
    feature_medians = X_all.median().to_dict()

    # ── OCCUPATION PROFILES FOR SKILL-GAP ANALYSIS ──
    skill_gap_cols = all_skill_cols + all_ability_cols + all_knowledge_cols
    valid_sgcols = [c for c in skill_gap_cols if c in merged.columns]
    occ_profiles = merged[["O*NET-SOC Code", "Title", "Description", "Job Zone"] + valid_sgcols].copy()
    for col in valid_sgcols:
        occ_profiles[col] = occ_profiles[col].fillna(occ_profiles[col].median())

    # ── SAVE ──
    print("\nSaving model artifacts...")
    artifact = {
        # Job Zone classifier
        "jz_pipeline": best_pipeline,
        "jz_model_name": best_name,
        "jz_model_results": results,
        # Occupation recommender
        "knn_recommender": knn_recommender,
        "recommender_scaler": scaler,
        "recommender_X": X_scaled,             # the occupation feature matrix (scaled)
        "recommender_titles": merged["Title"].tolist(),
        "recommender_codes": merged["O*NET-SOC Code"].tolist(),
        # Feature info
        "feature_cols": feature_cols,
        "feature_medians": feature_medians,
        # Occupation metadata
        "occ_profiles": occ_profiles,
        "all_skill_cols": all_skill_cols,
        "all_ability_cols": all_ability_cols,
        "all_knowledge_cols": all_knowledge_cols,
        # Job zone reference
        "jz_reference": {
            1: "Very Little Preparation",
            2: "Some Preparation",
            3: "Medium Preparation",
            4: "Considerable Preparation",
            5: "Extensive Preparation",
        },
    }

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved: {model_path}")
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"  Best JZ Classifier : {best_name}")
    print(f"  Features           : {len(feature_cols)}")
    print(f"  Occupations        : {len(merged)}")
    print(f"  Job Zone classes   : {sorted(pd.Series(y_jz).unique().tolist())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
