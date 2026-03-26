# 💳 Customer Credit Score Prediction

A Machine Learning web application that predicts a customer's credit score as **Good**, **Standard**, or **Bad** based on their financial behaviour and history.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://your-username-customer-credit-score-prediction.streamlit.app)

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Problem Type** | Supervised Learning — Multi-class Classification |
| **Target Column** | `Credit_Mix` → Good / Standard / Bad |
| **Dataset Size** | 50,000 rows · 27 features |
| **Best Model** | Random Forest |

---

## 🤖 Algorithms Used

| Algorithm | Accuracy |
|---|---|
| Logistic Regression | 63% |
| Support Vector Machine (SVM) | 67% |
| ✅ Random Forest *(Best)* | 82% |

---

## 📊 Input Features

| Feature | Description |
|---|---|
| Age | Customer age |
| Occupation | Job type |
| Annual Income | Yearly income |
| Monthly Inhand Salary | Take-home salary per month |
| Num Bank Accounts | Number of bank accounts |
| Num Credit Cards | Number of credit cards held |
| Interest Rate | Rate of interest on loans |
| Num of Loans | Total loans taken |
| Delay from Due Date | Average days delayed in payment |
| Num of Delayed Payments | Count of delayed payments |
| Outstanding Debt | Total outstanding debt |
| Credit Utilization Ratio | % of credit being used |
| Total EMI per Month | Monthly EMI amount |
| Amount Invested Monthly | Monthly investment amount |
| Monthly Balance | Remaining monthly balance |
| Payment of Min Amount | Whether minimum amount is paid |
| Payment Behaviour | Spending and payment pattern |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Array-lightblue?style=flat-square&logo=numpy)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-purple?style=flat-square)

---

## 📁 Project Structure

```
📁 customer-credit-score-prediction/
   ├── app.py                             # Streamlit web app
   ├── customer_credit_score_model.pkl    # Trained Random Forest model
   ├── scaler.pkl                         # StandardScaler
   ├── requirements.txt                   # Dependencies
   └── README.md                          # Project documentation
```

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your_username/customer-credit-score-prediction.git

# 2. Go into the folder
cd customer-credit-score-prediction

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📷 App Preview

> Fill in customer financial details → Click Predict → Get Credit Score instantly

| Result | Meaning |
|---|---|
| ✅ Good | Strong credit profile — low risk |
| ⚠️ Standard | Average credit profile — monitor |
| ❌ Bad | High risk credit profile — caution |

---

## 👩‍💻 Author

**Sneha**
B.Tech Student · Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-your_username-black?style=flat-square&logo=github)](https://github.com/your_username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your_profile)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
