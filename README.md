# CareerLens 🔭

**CareerLens** is a Machine Learning web application that analyzes your skill profile
and recommends the most suitable careers/occupations using the O\*NET occupational database.

## What it does

1. **Job Readiness Prediction** — Classifies your career readiness into one of 4 Job Zones (2–5) using a trained Random Forest classifier.
2. **Occupation Recommendation** — Finds the top 5 most similar occupations to your skill profile using KNN cosine similarity across 878 occupations.
3. **Skill-Match Analysis** — Shows which skills you already have for your top matched career.
4. **Skill-Gap Analysis** — Highlights the skills you are missing or need to improve.
5. **Learning Roadmap** — Prioritizes which skills to develop next, weighted by importance and gap size.

## Dataset

Uses the **O\*NET 29.1 database** (multi-table relational CSV files in `data/`):

| File | Role |
|------|------|
| `occupation_data.csv` | 1,016 occupations with titles and descriptions |
| `job_zones.csv` | Job Zone (1–5) per occupation — **classification target** |
| `career_interest_types.csv` | RIASEC interest scores (6 features) |
| `essential_skills.csv` | Essential skill importance scores (10 features selected) |
| `abilities.csv` | Cognitive/physical ability scores (10 features selected) |
| `knowledge.csv` | Knowledge domain scores (10 features selected) |
| `work_styles.csv` | Work style impact scores (8 features selected) |

**Total features:** 44 | **Occupations in model:** 878

## ML Models Compared

| Model | Test Accuracy | CV Accuracy |
|-------|---------------|-------------|
| **Random Forest** (best) | ~82% | ~80% |
| Gradient Boosting | ~80% | ~81% |
| K-Nearest Neighbors | ~82% | ~77% |

## Project Structure

```
CareerLens/
├── app.py                  # Streamlit web application
├── train_model.py          # ML training pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── agent_instructions.md   # Agent build instructions
├── .env.example            # Environment variable template
├── data/                   # O*NET CSV dataset files
    └──dataset.zip                  
└── models/
    └── model.pkl           # Trained model + artifacts
```

## Local Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
# or on Windows:
py train_model.py
```

This will create `models/model.pkl`.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub (include `data/` and `models/model.pkl`, or re-run training in the cloud).
2. Go to [share.streamlit.io](https://careerlenai.streamlit.app/).
3. Connect your GitHub repo.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

> **Note:** If `models/model.pkl` is not committed to the repo, add a startup command in Streamlit Cloud settings:
> ```
> python train_model.py
> ```
> Or add a `packages.txt` and the training step to `.streamlit/config.toml`.

## Environment Variables

No API keys are required. If you extend CareerLens with LLM features, use `.env` (see `.env.example`).

## Tech Stack

- **Python 3.9+**
- **pandas / numpy** — data loading and feature engineering
- **scikit-learn** — Random Forest, Gradient Boosting, KNN, StandardScaler, Pipeline
- **Streamlit** — web UI
- **openpyxl** — Excel file support (if needed for .xlsx variants)

---

Built as a portfolio/internship-ready ML application.
