import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.header-box {
    background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
    border-radius: 16px; padding: 32px 36px;
    margin-bottom: 28px; color: white;
}
.header-box h1 { font-size: 28px; font-weight: 600; margin: 0 0 6px 0; color: white; }
.header-box p  { font-size: 14px; color: #a0aec0; margin: 0; }
.section-title {
    font-size: 13px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #718096; margin: 24px 0 10px 0;
}
.result-good     { background:#e6f9f0; border-left:5px solid #38a169; border-radius:12px; padding:24px 28px; }
.result-standard { background:#fffbeb; border-left:5px solid #d69e2e; border-radius:12px; padding:24px 28px; }
.result-bad      { background:#fff5f5; border-left:5px solid #e53e3e; border-radius:12px; padding:24px 28px; }
.result-label { font-size:13px; color:#718096; margin-bottom:4px; font-weight:500; }
.result-value { font-size:32px; font-weight:700; margin:0; }
.result-good     .result-value { color:#276749; }
.result-standard .result-value { color:#975a16; }
.result-bad      .result-value { color:#9b2c2c; }
.result-desc { font-size:13px; margin-top:8px; color:#4a5568; }
.prob-row { display:flex; justify-content:space-between; align-items:center; margin:8px 0; }
.prob-label { font-size:13px; color:#4a5568; width:80px; }
.prob-bar-bg { flex:1; height:8px; background:#e2e8f0; border-radius:4px; margin:0 12px; }
.prob-bar { height:8px; border-radius:4px; }
.prob-val { font-size:13px; font-weight:600; color:#2d3748; width:42px; text-align:right; }
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #2d3561, #4a5568);
    color: white; border: none; border-radius: 10px;
    padding: 12px 32px; font-size: 15px; font-weight: 600;
    font-family: 'DM Sans', sans-serif; width: 100%; cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>💳 Credit Score Predictor</h1>
    <p>Fill in the customer details below to predict their credit score.</p>
</div>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load("customer_credit_score_model.pkl")
    scaler     = joblib.load("scaler.pkl")
    train_cols = joblib.load("train_columns.pkl")
    return model, scaler, train_cols

le = LabelEncoder()
le.classes_ = np.array(['Bad', 'Good', 'Standard'])

try:
    model, scaler, train_cols = load_model()
    model_loaded  = True
except:
    model_loaded  = False
    st.error("❌ Model files not found. Make sure all .pkl files are in the same folder as app.py")

# ── Input Form ─────────────────────────────────────────────
st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age        = st.number_input("Age", min_value=18, max_value=100, value=30)
    occupation = st.selectbox("Occupation", [
        'Scientist','Teacher','Engineer','Entrepreneur','Developer',
        'Lawyer','Media_Manager','Doctor','Journalist','Manager',
        'Accountant','Musician','Mechanic','Writer','Architect'
    ])
with col2:
    annual_income         = st.number_input("Annual Income (₹)", min_value=0, value=50000, step=1000)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary (₹)", min_value=0, value=4000, step=100)

st.markdown('<div class="section-title">Credit & Loan Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    num_bank_accounts = st.number_input("No. of Bank Accounts", min_value=0, max_value=20,  value=3)
    num_credit_card   = st.number_input("No. of Credit Cards",  min_value=0, max_value=15,  value=3)
    num_of_loan       = st.number_input("No. of Loans",         min_value=0, max_value=20,  value=2)
    interest_rate     = st.number_input("Interest Rate (%)",    min_value=0, max_value=50,  value=12)
with col4:
    outstanding_debt         = st.number_input("Outstanding Debt (₹)",    min_value=0.0, value=800.0, step=50.0)
    credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 28.5)
    total_emi_per_month      = st.number_input("Total EMI / Month (₹)",  min_value=0.0, value=150.0, step=10.0)
    num_credit_inquiries     = st.number_input("No. of Credit Inquiries", min_value=0, max_value=20, value=3)

st.markdown('<div class="section-title">Payment Behaviour</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    delay_from_due_date    = st.slider("Delay from Due Date (days)", 0, 67, 5)
    num_of_delayed_payment = st.number_input("No. of Delayed Payments", min_value=0, max_value=30, value=2)
    payment_of_min_amount  = st.selectbox("Pays Minimum Amount?", ['Yes', 'No', 'NM'])
with col6:
    payment_behaviour       = st.selectbox("Payment Behaviour", [
        'Low_spent_Small_value_payments',
        'Low_spent_Medium_value_payments',
        'Low_spent_Large_value_payments',
        'High_spent_Small_value_payments',
        'High_spent_Medium_value_payments',
        'High_spent_Large_value_payments'
    ])
    changed_credit_limit    = st.number_input("Changed Credit Limit",         min_value=0.0, value=2.5,  step=0.5)
    amount_invested_monthly = st.number_input("Amount Invested Monthly (₹)", min_value=0.0, value=200.0, step=10.0)
    monthly_balance         = st.number_input("Monthly Balance (₹)",         min_value=0.0, value=300.0, step=10.0)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Credit Score")

# ── Prediction ─────────────────────────────────────────────
if predict_btn and model_loaded:

    input_dict = {
        'Age'                     : age,
        'Annual_Income'           : annual_income,
        'Monthly_Inhand_Salary'   : monthly_inhand_salary,
        'Num_Bank_Accounts'       : num_bank_accounts,
        'Num_Credit_Card'         : num_credit_card,
        'Interest_Rate'           : interest_rate,
        'Num_of_Loan'             : num_of_loan,
        'Delay_from_due_date'     : delay_from_due_date,
        'Num_of_Delayed_Payment'  : num_of_delayed_payment,
        'Changed_Credit_Limit'    : changed_credit_limit,
        'Num_Credit_Inquiries'    : num_credit_inquiries,
        'Outstanding_Debt'        : outstanding_debt,
        'Credit_Utilization_Ratio': credit_utilization_ratio,
        'Total_EMI_per_month'     : total_emi_per_month,
        'Amount_invested_monthly' : amount_invested_monthly,
        'Monthly_Balance'         : monthly_balance,
        'Payment_of_Min_Amount'   : 1 if payment_of_min_amount == 'Yes' else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode Occupation
    for occ in ['Scientist','Teacher','Engineer','Entrepreneur','Developer','Lawyer',
                'Media_Manager','Doctor','Journalist','Manager','Accountant',
                'Musician','Mechanic','Writer','Architect']:
        input_df[f'Occupation_{occ}'] = 1 if occupation == occ else 0

    # One-hot encode Payment_Behaviour
    for pb in ['Low_spent_Small_value_payments','Low_spent_Medium_value_payments',
               'Low_spent_Large_value_payments','High_spent_Small_value_payments',
               'High_spent_Medium_value_payments','High_spent_Large_value_payments']:
        input_df[f'Payment_Behaviour_{pb}'] = 1 if payment_behaviour == pb else 0

    try:
        input_df     = input_df.reindex(columns=train_cols, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        label        = le.inverse_transform([prediction])[0]
        classes      = le.classes_

        css_class  = {'Good':'result-good','Standard':'result-standard','Bad':'result-bad'}.get(label,'result-standard')
        emoji      = {'Good':'✅','Standard':'⚠️','Bad':'❌'}.get(label,'')
        desc       = {
            'Good'    : 'This customer has a strong credit profile with low risk.',
            'Standard': 'This customer has an average credit profile. Monitor key metrics.',
            'Bad'     : 'This customer has a high-risk credit profile. Caution advised.'
        }.get(label,'')
        bar_colors = {'Good':'#38a169','Standard':'#d69e2e','Bad':'#e53e3e'}

        prob_bars = ""
        for cls, prob in zip(classes, proba):
            color = bar_colors.get(cls, '#718096')
            prob_bars += f"""
            <div class="prob-row">
                <span class="prob-label">{cls}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar" style="width:{prob*100:.1f}%;background:{color}"></div>
                </div>
                <span class="prob-val">{prob*100:.1f}%</span>
            </div>"""

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">Predicted Credit Score</div>
            <p class="result-value">{emoji} {label}</p>
            <div class="result-desc">{desc}</div>
            <div style="margin-top:16px">
                <strong style="font-size:12px;color:#718096;letter-spacing:0.06em;text-transform:uppercase">Confidence</strong>
                {prob_bars}
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── Footer ─────────────────────────────────────────────────
st.markdown("<br><hr style='border-color:#e2e8f0'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a0aec0;font-size:12px'>Credit Score Predictor · Random Forest Model</p>", unsafe_allow_html=True)
