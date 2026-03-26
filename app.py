import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.header-box {
    background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
    border-radius: 16px; padding: 32px 36px; margin-bottom: 28px; color: white;
}
.header-box h1 { font-size: 28px; font-weight: 600; margin: 0 0 6px 0; color: white; }
.header-box p  { font-size: 14px; color: #a0aec0; margin: 0; }
.section-title {
    font-size: 13px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #718096; margin: 24px 0 10px 0;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #2d3561, #4a5568);
    color: white; border: none; border-radius: 10px;
    padding: 12px 32px; font-size: 15px; font-weight: 600;
    font-family: 'DM Sans', sans-serif; width: 100%; cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>💳 Credit Score Predictor</h1>
    <p>Fill in the customer details below to predict their credit score.</p>
</div>
""", unsafe_allow_html=True)

# Load model — NO cache so it always loads fresh
model      = joblib.load('customer_credit_score_model.pkl')
scaler     = joblib.load('scaler.pkl')
train_cols = joblib.load('train_columns.pkl')

le = LabelEncoder()
le.classes_ = np.array(['Bad', 'Good', 'Standard'])

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
    num_bank_accounts = st.number_input("No. of Bank Accounts", min_value=0, max_value=20, value=3)
    num_credit_card   = st.number_input("No. of Credit Cards",  min_value=0, max_value=15, value=3)
    num_of_loan       = st.number_input("No. of Loans",         min_value=0, max_value=20, value=2)
    interest_rate     = st.number_input("Interest Rate (%)",    min_value=0, max_value=50, value=12)
with col4:
    outstanding_debt         = st.number_input("Outstanding Debt (₹)",    min_value=0.0, value=800.0,  step=50.0)
    credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 28.5)
    total_emi_per_month      = st.number_input("Total EMI / Month (₹)",  min_value=0.0, value=150.0,  step=10.0)
    num_credit_inquiries     = st.number_input("No. of Credit Inquiries", min_value=0,   max_value=20, value=3)

st.markdown('<div class="section-title">Payment Behaviour</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    delay_from_due_date    = st.slider("Delay from Due Date (days)", 0, 67, 5)
    num_of_delayed_payment = st.number_input("No. of Delayed Payments", min_value=0, max_value=30, value=2)
    payment_of_min_amount  = st.selectbox("Pays Minimum Amount?", ['Yes', 'No', 'NM'])
with col6:
    payment_behaviour       = st.selectbox("Payment Behaviour", [
        'Low_spent_Small_value_payments','Low_spent_Medium_value_payments',
        'Low_spent_Large_value_payments','High_spent_Small_value_payments',
        'High_spent_Medium_value_payments','High_spent_Large_value_payments'
    ])
    changed_credit_limit    = st.number_input("Changed Credit Limit",         min_value=0.0, value=2.5,   step=0.5)
    amount_invested_monthly = st.number_input("Amount Invested Monthly (₹)", min_value=0.0, value=200.0, step=10.0)
    monthly_balance         = st.number_input("Monthly Balance (₹)",         min_value=0.0, value=300.0, step=10.0)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Credit Score")

if predict_btn:

    input_dict = {
        'Age'                     : int(age),
        'Annual_Income'           : float(annual_income),
        'Monthly_Inhand_Salary'   : float(monthly_inhand_salary),
        'Num_Bank_Accounts'       : int(num_bank_accounts),
        'Num_Credit_Card'         : int(num_credit_card),
        'Interest_Rate'           : int(interest_rate),
        'Num_of_Loan'             : int(num_of_loan),
        'Delay_from_due_date'     : int(delay_from_due_date),
        'Num_of_Delayed_Payment'  : int(num_of_delayed_payment),
        'Changed_Credit_Limit'    : float(changed_credit_limit),
        'Num_Credit_Inquiries'    : int(num_credit_inquiries),
        'Outstanding_Debt'        : float(outstanding_debt),
        'Credit_Utilization_Ratio': float(credit_utilization_ratio),
        'Total_EMI_per_month'     : float(total_emi_per_month),
        'Amount_invested_monthly' : float(amount_invested_monthly),
        'Monthly_Balance'         : float(monthly_balance),
        'Payment_of_Min_Amount'   : 1 if payment_of_min_amount == 'Yes' else 0,
    }

    input_df = pd.DataFrame([input_dict])

    for occ in ['Scientist','Teacher','Engineer','Entrepreneur','Developer','Lawyer',
                'Media_Manager','Doctor','Journalist','Manager','Accountant',
                'Musician','Mechanic','Writer','Architect']:
        input_df[f'Occupation_{occ}'] = 1 if occupation == occ else 0

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

        colors = {'Good': '#38a169', 'Standard': '#d69e2e', 'Bad': '#e53e3e'}
        bg     = {'Good': '#e6f9f0', 'Standard': '#fffbeb', 'Bad': '#fff5f5'}
        emoji  = {'Good': '✅',       'Standard': '⚠️',      'Bad': '❌'}
        desc   = {
            'Good'    : 'This customer has a strong credit profile with low risk.',
            'Standard': 'This customer has an average credit profile. Monitor key metrics.',
            'Bad'     : 'This customer has a high-risk credit profile. Caution advised.'
        }

        st.markdown(f"""
        <div style="background:{bg[label]};border-left:5px solid {colors[label]};
                    border-radius:12px;padding:24px 28px;margin-top:16px;">
            <div style="font-size:13px;color:#718096;font-weight:500;">Predicted Credit Score</div>
            <div style="font-size:32px;font-weight:700;color:{colors[label]};margin:4px 0;">
                {emoji[label]} {label}
            </div>
            <div style="font-size:13px;color:#4a5568;margin-top:4px;">{desc[label]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**Confidence**", unsafe_allow_html=True)
        for cls, prob in zip(classes, proba):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.progress(float(prob), text=cls)
            with col_b:
                st.markdown(f"**{prob*100:.1f}%**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("<br><hr style='border-color:#e2e8f0'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a0aec0;font-size:12px'>Credit Score Predictor · Random Forest Model</p>", unsafe_allow_html=True)
