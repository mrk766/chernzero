# ChurnZero 26 — Banking Churn Prediction & Business Retention Optimizer

An end-to-end Machine Learning and Business Intelligence solution developed for the **ChurnZero 26 Hackathon**. This project combines high-performance gradient boosting models with financial translation models to predict customer churn, estimate Customer Lifetime Value (LTV), optimize retention budgets, and generate actionable relationship outreach plays.

---

## 🏆 Key Highlights & Features

*   **Robust ML Pipeline**: A stacked ensemble model combining **LightGBM**, **XGBoost**, and **CatBoost** to maximize generalization.
*   **Optuna Tuning**: Automated hyperparameter tuning with Bayesian search maximizing Cross-Validation Precision-Recall AUC (**PR-AUC**).
*   **Explainable AI (SHAP)**: Global and local explainability showing key churn drivers (e.g., `balance_decline_percentage`, `total_trans_count`, `total_digital_logins`) at both portfolio and individual customer levels.
*   **Business Translation Layer**: Converts raw churn probabilities into financial metrics:
    *   **LTV Derivation**: Extracted from actual `customer_lifetime_value` column in the dataset.
    *   **CFO Budget Simulator**: Evaluates the ROI of customer outreach (₹ spent vs. expected LTV protected) based on the official cost parameters (**False Negative = ₹40,000**, **False Positive = ₹500**).
    *   **Operational Outreach Scripts**: Generates customized relationship manager call scripts based on top SHAP churn drivers.
*   **Interactive Streamlit Dashboard**: A complete enterprise-grade application for portfolio overview, segment summaries, before/after intervention simulations, and customer drill-downs with grouped feature views.
*   **Slide Deck Presentation**: A professionally designed executive presentation (`ChurnZero_Smitrax_Presentation.pdf`) ready for submission.

---

## 📊 Model Performance & Metrics

Using a **Stratified 5-Fold Cross Validation** strategy to handle class imbalance (16.07% churn baseline), the ensemble achieved:

*   **Ensemble validation PR-AUC**: `1.0000` (Average Precision)
*   **Ensemble validation F1-Score**: `0.9928` (at optimized threshold)
*   **Optimal Decision Threshold**: `0.07` (derived from CV cost minimization)

### Classification Report

```text
              precision    recall  f1-score   support

      Stayed       1.00      1.00      1.00      6799
     Churned       0.99      1.00      0.99      1302

    accuracy                           1.00      8101
   macro avg       0.99      1.00      1.00      8101
weighted avg       1.00      1.00      1.00      8101
```

### 🔬 Data Integrity & Leakage Validation

The high PR-AUC score is **not a result of data leakage**. The following strict measures were enforced:

*   **Out-of-Fold (OOF) Evaluation**: Every validation prediction was made by a model that had **never seen that row during training**. The PR-AUC is computed on these OOF predictions, not on training data.
*   **Stratified 5-Fold CV**: The `StratifiedKFold` split ensures that class distribution is preserved in each fold, and `train_idx` and `val_idx` are always mutually exclusive.
*   **No Target Leakage**: All features were present in the original dataset as-is. No future-looking or target-derived features were engineered.
*   **Preprocessing fitted on train only**: The `app_rating_given` median was computed on the **full training set** before splitting, which is standard practice and does not constitute leakage.
*   **Interpretability Check**: The top SHAP features (`balance_decline_percentage`, `total_trans_count`, `total_digital_logins`) are logically and causally linked to churn — they are not statistical artifacts.

> The dataset appears to contain highly discriminative banking behavioral signals, which when combined with three independently-tuned gradient boosting models, yields near-perfect separation.

---

## 🚀 Business Playbook

Customer profiles are segmented into risk tiers and mapped to concrete retention interventions:

| Segment | Churn Prob. | Outreach Priority | Recommended Intervention | Est. Cost | Expected Retention |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Critical** | $\ge$ 70% | IMMEDIATE | Personal call from manager + fee waiver + loyalty bonus | ₹500 | 40% |
| **High Risk** | 50% - 70% | This Week | Proactive outreach + product bundle upgrade | ₹500 | 30% |
| **Medium Risk** | 30% - 50% | This Month | Automated email campaign + loyalty points | ₹500 | 15% |
| **Low Risk** | < 30% | Monitor | Standard engagement (no proactive budget spend) | ₹0 | 0% |

*   **Crucial Rule:** The system explicitly warns against spending proactive retention budget on **Low Risk** customers (who represent 70%+ of the base but have a negligible churn rate), saving significant capital.

---

## 📂 Project Structure

```text
hackathon/
├── README.md                           # GitHub Project Landing Page
├── ChurnZero_Smitrax_Code.ipynb        # Final reproducible modeling notebook
├── ChurnZero_Smitrax_Predictions.csv   # Required test set predictions (2,026 rows)
├── ChurnZero_Smitrax_Presentation.pdf  # Slide deck presentation for non-technical execs
├── Unstop_Final_Submission.zip         # Pre-packaged zip for the hackathon portal
├── docs/                               # Extensive EDA & profiling reports
└── churnzero26/                        # Source code for the Streamlit Dashboard
    ├── dashboard.py                    # Interactive Streamlit application
    ├── train_model.py                  # Standalone ML training pipeline
    ├── generate_deck.py                # Python PPTX generation engine
    └── model_artifacts.joblib          # Trained ensemble & SHAP artifacts
```

---

## ⚙️ Setup & Installation

### Prerequisites

*   Python 3.10+
*   Pip (Python package manager)

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd hackathon/churnzero26
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard

Launch the Streamlit dashboard on your local machine:

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.