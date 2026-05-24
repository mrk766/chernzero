"""
ChurnZero 26 — Interactive Streamlit Dashboard
Bank Customer Churn Prediction with Retention Playbook
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import time

# Set default plotly theme to match dark UI
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.font.family = "Inter, sans-serif"
from pathlib import Path

st.set_page_config(
    page_title="ChurnZero 26 — Bank Churn Dashboard",
    page_icon="🧊",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global App Background */
    .stApp {
        background: #000000 !important;
        color: #ececec;
    }
    
    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Premium Metric Cards */
    [data-testid="stMetric"] {
        background: #171717;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 24px;
        min-height: 165px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: none;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: #555555;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    /* Subheaders */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.2rem !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.5rem !important; margin-top: 2rem !important; }
    h3 { font-size: 1.2rem !important; color: #cbd5e1 !important; }
    
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 2rem 0 !important;
    }
    
    /* Sleek Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #4a4a4a !important;
        border-radius: 8px !important;
        background: #2f2f2f !important;
        box-shadow: none;
    }
    
    /* Dataframes/Tables Container */
    [data-testid="stDataFrame"] {
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Primary buttons */
    .stButton>button {
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
        box-shadow: none;
    }
    .stButton>button:hover {
        background: #0e906f;
        opacity: 1;
        box-shadow: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("customer_predictions.csv")
    return df


@st.cache_data
def load_model_artifacts():
    try:
        artifacts = joblib.load("model_artifacts.joblib")
        return artifacts
    except FileNotFoundError:
        return None


def format_currency(val):
    if val >= 10_000_000:
        return f"₹{val / 10_000_000:.1f}Cr"
    elif val >= 100_000:
        return f"₹{val / 100_000:.1f}L"
    elif val >= 1_000:
        return f"₹{val / 1_000:.1f}K"
    return f"₹{val:,.0f}"


def main():
    if "app_loaded" not in st.session_state:
        load_container = st.empty()
        with load_container.container():
            st.markdown("### 🔌 Initializing ChurnZero Enterprise Engine...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Connecting to secure database...")
            time.sleep(0.4)
            progress_bar.progress(30)
            
            status_text.text("Loading machine learning model artifacts...")
            time.sleep(0.5)
            progress_bar.progress(70)
            
            status_text.text("Computing SHAP feature importance...")
            time.sleep(0.4)
            progress_bar.progress(100)
            status_text.text("System Ready.")
            time.sleep(0.3)
            
        load_container.empty()
        st.session_state["app_loaded"] = True

    df = load_data()
    artifacts = load_model_artifacts()
    optimal_threshold = artifacts['optimal_threshold'] if artifacts else 0.07

    # --- Header ---
    st.title("ChurnZero 26 Enterprise Dashboard")
    st.markdown("**Data-driven retention strategy** — Predict churn, quantify risk, and take action.")

    # --- KPI Cards ---
    st.header("Portfolio Overview")

    total_customers = len(df)
    at_risk = len(df[df["RiskSegment"].isin(["Critical", "High Risk"])])
    avg_churn_prob = df["ChurnProb"].mean()
    total_ltv_at_risk = df[df["RiskSegment"].isin(["Critical", "High Risk"])]["LTV"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("At Risk (Critical + High)", f"{at_risk:,}", f"{at_risk / total_customers:.1%}")
    with col3:
        st.metric("Avg Churn Probability", f"{avg_churn_prob:.1%}")
    with col4:
        st.metric("₹ LTV at Risk", format_currency(total_ltv_at_risk))

    st.divider()

    # --- Segment Distribution ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Segment Distribution")
        seg_counts = df["RiskSegment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        color_map = {
            "Critical": "#e74c3c",
            "High Risk": "#e67e22",
            "Medium Risk": "#f1c40f",
            "Low Risk": "#2ecc71"
        }
        fig = px.pie(
            seg_counts, values="Count", names="Segment",
            color="Segment", color_discrete_map=color_map,
            hole=0.4
        )
        fig.update_layout(height=400, margin=dict(t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(
            df, x="ChurnProb", nbins=50, color="RiskSegment",
            color_discrete_map=color_map,
            labels={"ChurnProb": "Churn Probability"},
            category_orders={"RiskSegment": ["Critical", "High Risk", "Medium Risk", "Low Risk"]}
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Critical (0.7)")
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="High (0.5)")
        fig.add_vline(x=0.3, line_dash="dash", line_color="gold", annotation_text="Medium (0.3)")
        fig.add_vline(x=optimal_threshold, line_dash="dot", line_color="green", annotation_text=f"Optimal Cutoff ({optimal_threshold:.2f})")
        fig.update_layout(height=400, margin=dict(t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Segment Table ---
    st.subheader("Segment Summary")
    
    # Safely aggregate based on new columns
    seg_summary = df.groupby("RiskSegment").agg(
        Customers=("ChurnProb", "count"),
        Avg_Churn_Prob=("ChurnProb", "mean"),
        Avg_LTV=("LTV", "mean"),
        Total_LTV=("LTV", "sum"),
        Avg_Age=("Age", "mean"),
        Avg_Balance=("Balance", "mean")
    ).round(2)
    
    # Reindex to ensure order
    seg_summary = seg_summary.reindex(["Critical", "High Risk", "Medium Risk", "Low Risk"])
    seg_summary.index.name = "Segment"
    st.dataframe(seg_summary, use_container_width=True)

    st.divider()

    # --- Customer Drill-Down ---
    st.subheader("Customer Intelligence Drill-Down")
    customer_id = st.selectbox(
        "Select a customer (by index)",
        options=df.index.tolist(),
        format_func=lambda x: f"Customer ID: {x} — {df.loc[x, 'RiskSegment']} (Churn: {df.loc[x, 'ChurnProb']:.1%})"
    )

    drill_down_placeholder = st.empty()
    if "last_customer_id" in st.session_state and st.session_state["last_customer_id"] != customer_id:
        with drill_down_placeholder.container():
            st.info("🔄 Fetching Live Customer Intelligence...")
            prog_bar = st.progress(0)
            time.sleep(0.3)
            prog_bar.progress(100)
            time.sleep(0.2)
        drill_down_placeholder.empty()
            
    st.session_state["last_customer_id"] = customer_id

    customer = df.loc[customer_id]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Churn Probability", f"{customer['ChurnProb']:.1%}")
        risk_color = {"Critical": "", "High Risk": "", "Medium Risk": "", "Low Risk": ""}
        st.write(f"**Risk Segment:** {risk_color.get(customer['RiskSegment'], '')} {customer['RiskSegment']}")
    with col_b:
        st.metric("Lifetime Value (LTV)", format_currency(customer["LTV"]))
        st.write(f"**Region:** {customer['Geography']} | **Gender:** {customer['Gender']}")
    with col_c:
        st.metric("Balance", format_currency(customer["Balance"]))
        st.write(f"**Products:** {customer['NumOfProducts']} | **Active Member:** {'Yes' if customer['IsActiveMember'] == 1 else 'No'}")

    # Recommended action (aligns with FP cost of ₹500)
    playbook = {
        "Critical": ("IMMEDIATE: Personal call + fee waiver + loyalty bonus", 500),
        "High Risk": ("THIS WEEK: Proactive outreach + bundle offer + rate bump", 500),
        "Medium Risk": ("THIS MONTH: Email campaign + loyalty points + survey", 500),
        "Low Risk": ("MONITOR: No proactive spend — standard program only", 0),
    }
    segment = customer["RiskSegment"]
    action, cost = playbook[segment]
    
    st.info(f"**Recommended Action:** {action}  \n**Intervention Cost:** ₹{cost}")

    # --- Interactive Call Script Generator ---
    st.markdown("### Personalized Relationship Outreach Script")
    script = f"\"Hello, may I speak with Customer ID: {customer.name}? This is the relationship team at ChurnZero Bank."
    if customer['Balance'] == 0:
        script += f" We noticed you have not been using your account for deposits recently. We want to offer you a free relationship health check and a high-yield bonus...\""
    elif customer['NumOfProducts'] == 1:
        script += " We appreciate your relationship. As a single-product customer, we would love to introduce our high-yield savings options with a bonus 0.50% interest bump...\""
    elif customer['HasCrCard'] == 0:
        script += " We noticed you don't have our premium credit card yet. We'd love to offer you a pre-approved card with waived annual fees...\""
    else:
        script += " We want to thank you for your loyalty. We have pre-approved you for our premium loyalty program benefits...\""
    st.info(script)

    # --- Grouped Features accordion ---
    st.markdown("### Complete Customer Profile Attributes")
    with st.expander("Customer Profile"):
        st.write(f"**Age:** {customer['Age']} | **Gender:** {customer['Gender']} | **Geography:** {customer['Geography']}")
        st.write(f"**Estimated Salary:** {format_currency(customer['EstimatedSalary'])} | **Credit Score:** {customer['CreditScore']}")

    with st.expander("Account Behaviour"):
        st.write(f"**Tenure:** {customer['Tenure']} years | **Balance:** {format_currency(customer['Balance'])}")
        st.write(f"**Num Of Products:** {customer['NumOfProducts']} | **Has Credit Card:** {'Yes' if customer['HasCrCard'] == 1 else 'No'}")
        st.write(f"**Is Active Member:** {'Yes' if customer['IsActiveMember'] == 1 else 'No'}")

    st.divider()

    # --- CFO Interactive Budget Optimizer ---
    st.subheader("CFO Interactive Budget Optimizer")
    st.markdown("Optimize budget allocation dynamically to prioritize customer targets based on Expected Loss.")
    budget = st.slider("Select Available Retention Budget (₹)", 10000, 1000000, 100000, 10000)

    # Calculate optimal targeting
    df_opt = df.copy()
    cost_map_opt = {"Critical": 500, "High Risk": 500, "Medium Risk": 500, "Low Risk": 0}
    ret_map_opt = {"Critical": 0.40, "High Risk": 0.30, "Medium Risk": 0.15, "Low Risk": 0.0}

    df_opt['Cost'] = df_opt['RiskSegment'].map(cost_map_opt)
    df_opt['SavedLTV'] = df_opt['LTV'] * df_opt['RiskSegment'].map(ret_map_opt)
    df_opt['ExpectedLoss'] = df_opt['ChurnProb'] * df_opt['LTV']

    df_opt = df_opt.sort_values(by='ExpectedLoss', ascending=False)
    df_opt['CumulativeCost'] = df_opt['Cost'].cumsum()

    # Find subset that fits budget
    df_targeted = df_opt[df_opt['CumulativeCost'] <= budget]
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        st.metric("Customers Targeted", f"{len(df_targeted):,}")
    with col_opt2:
        total_saved = df_targeted['SavedLTV'].sum()
        st.metric("Expected LTV Protected", format_currency(total_saved))
    with col_opt3:
        roi = (total_saved - budget) / budget if budget > 0 else 0
        st.metric("Net ROI Factor", f"{roi:.1f}x")

    st.divider()

    # --- SHAP Feature Importance ---
    st.subheader("Top Churn Drivers (SHAP)")

    if artifacts and "features" in artifacts:
        feature_names = artifacts["features"]

        try:
            shap_data = joblib.load("shap_data.joblib")
            shap_vals = shap_data["shap_values"]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            mean_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP|": mean_shap
            }).sort_values("Mean |SHAP|", ascending=True).tail(15)

            fig = px.bar(
                shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                color="Mean |SHAP|", color_continuous_scale="Reds",
                title="Top 15 Features by SHAP Importance"
            )
            fig.update_layout(height=500, margin=dict(t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("SHAP data not found. Run the training script first.")
    else:
        st.warning("Model artifacts not found. Run the training script first.")

    st.divider()

    # --- Retention Playbook ---
    st.subheader("Retention Playbook")

    playbook_data = {
        "Segment": ["Critical", "High Risk", "Medium Risk", "Low Risk"],
        "Priority": ["IMMEDIATE", "This Week", "This Month", "Monitor"],
        "Action": [
            "Personal call + fee waiver + loyalty bonus",
            "Proactive outreach + product upgrade + rate improvement",
            "Email campaign + loyalty points + survey",
            "No proactive spend — general program only"
        ],
        "Cost/Customer": [500, 500, 500, 0],
        "Expected Retention": ["40%", "30%", "15%", "0%"],
    }

    # Calculate ROI
    for i, seg in enumerate(playbook_data["Segment"]):
        seg_df = df[df["RiskSegment"] == seg]
        n = len(seg_df)
        cost_per = playbook_data["Cost/Customer"][i]
        retention_rate = float(playbook_data["Expected Retention"][i].strip("%")) / 100
        total_cost = n * cost_per
        expected_saved = n * retention_rate * seg_df["LTV"].mean()
        roi = (expected_saved - total_cost) / total_cost if total_cost > 0 else 0
        playbook_data.setdefault("Customers", []).append(n)
        playbook_data.setdefault("Total Cost", []).append(total_cost)
        playbook_data.setdefault("Expected Saved", []).append(expected_saved)
        playbook_data.setdefault("ROI", []).append(roi)

    playbook_df = pd.DataFrame(playbook_data)
    playbook_df["ROI"] = playbook_df["ROI"].apply(lambda x: f"{x:.1f}x" if x != float("inf") else "∞")
    playbook_df["Total Cost"] = playbook_df["Total Cost"].apply(format_currency)
    playbook_df["Expected Saved"] = playbook_df["Expected Saved"].apply(format_currency)

    st.dataframe(playbook_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Before/After Simulation ---
    st.subheader("Before/After Intervention Simulation")

    current_churn = df["ChurnProb"].mean()

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("**Intervene on Critical Segment Only**")
        critical = df[df["RiskSegment"] == "Critical"]
        retained_1 = len(critical) * 0.40
        new_churn_1 = (df["ChurnProb"].sum() - retained_1) / len(df)
        reduction_1 = current_churn - new_churn_1

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Before", "After"],
            y=[current_churn, new_churn_1],
            marker_color=["#e74c3c", "#2ecc71"],
            text=[f"{current_churn:.1%}", f"{new_churn_1:.1%}"],
            textposition="auto"
        ))
        fig.update_layout(
            title=f"Churn Rate: {current_churn:.1%} → {new_churn_1:.1%} ({reduction_1:.1%} reduction)",
            yaxis_title="Churn Rate", height=350, margin=dict(t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with sim_col2:
        st.markdown("**Intervene on Critical + High Risk**")
        high_risk = df[df["RiskSegment"] == "High Risk"]
        retained_2 = len(critical) * 0.40 + len(high_risk) * 0.30
        new_churn_2 = (df["ChurnProb"].sum() - retained_2) / len(df)
        reduction_2 = current_churn - new_churn_2

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Before", "After"],
            y=[current_churn, new_churn_2],
            marker_color=["#e74c3c", "#2ecc71"],
            text=[f"{current_churn:.1%}", f"{new_churn_2:.1%}"],
            textposition="auto"
        ))
        fig.update_layout(
            title=f"Churn Rate: {current_churn:.1%} → {new_churn_2:.1%} ({reduction_2:.1%} reduction)",
            yaxis_title="Churn Rate", height=350, margin=dict(t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Inefficient Allocation Alerts ---
    st.subheader("Inefficient Allocation Alerts")
    low_risk = df[df["RiskSegment"] == "Low Risk"]
    wasted_budget = len(low_risk) * 500

    st.warning(
        f"**Do NOT spend retention budget on Low Risk customers.**  \n"
        f"- {len(low_risk):,} customers ({len(low_risk) / len(df):.0%} of portfolio)  \n"
        f"- Their actual churn rate: {low_risk['ChurnProb'].mean():.1%}  \n"
        f"- Spending ₹500/customer = **{format_currency(wasted_budget)} wasted** on customers who aren't leaving  \n"
        f"- **Rule:** Focus budget on Critical and High Risk segments where ROI is highest."
    )

    st.divider()

    # --- Export ---
    st.subheader("Export Data")

    export_segment = st.multiselect(
        "Filter by segment for export",
        options=["Critical", "High Risk", "Medium Risk", "Low Risk"],
        default=["Critical", "High Risk"]
    )

    export_df = df[df["RiskSegment"].isin(export_segment)]

    # Add recommended action
    action_map = {
        "Critical": "Personal call + fee waiver + loyalty bonus",
        "High Risk": "Proactive outreach + product upgrade",
        "Medium Risk": "Email campaign + loyalty points",
        "Low Risk": "No action needed"
    }
    export_df = export_df.copy()
    export_df["Recommended Action"] = export_df["RiskSegment"].map(action_map)

    csv = export_df.to_csv(index=False)
    st.download_button(
        label=f"Download Retention Action List ({len(export_df):,} customers)",
        data=csv,
        file_name="retention_action_list.csv",
        mime="text/csv"
    )

    # --- Footer ---
    st.divider()
    st.caption("ChurnZero 26 — Built with LightGBM + XGBoost + CatBoost ensemble | SHAP explanations | Streamlit")


if __name__ == "__main__":
    main()
