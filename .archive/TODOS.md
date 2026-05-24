# TODOS — ChurnZero 26

## Full Interactive Dashboard
- **What:** Build a Streamlit dashboard with customer drill-down, SHAP explanations, segment overview, and retention playbook
- **Why:** Judges remember what they can click. Most teams submit notebooks only — a dashboard stands out and looks deployable.
- **Pros:** Interactive demo is rare in hackathons, memorable for judges, looks business-ready
- **Cons:** Takes 4-6 hours, could squeeze model tuning time
- **Context:** Components: KPI cards (total customers, at risk, avg churn prob, $ at risk), segment distribution chart, SHAP feature importance, customer drill-down (select ID → risk score + SHAP force plot + recommended action), retention playbook table with filters.
- **Depends on:** Trained model (Step 3-4), business translation (Step 5)

## Inline Notebook Assertions
- **What:** Add shape checks, NaN guards, and convergence warnings after each major cell
- **Why:** Silent failures waste debugging time in a 24-48h hackathon. A shape mismatch after feature engineering is invisible without assertions.
- **Pros:** Catches bugs early, saves 30-60 min of debugging, makes notebook self-documenting
- **Cons:** Extra code in notebook (minimal), slight noise in output
- **Context:** After each transform (feature engineering, model training, evaluation), assert expected shapes, check for NaN/inf, verify model converged. Use `assert` statements or `pd.DataFrame.shape` checks.
- **Depends on:** Dataset must be loaded first (Step 1)
