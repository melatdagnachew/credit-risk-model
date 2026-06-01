# Credit Scoring Business Understanding

## 1. Basel II and the Need for Interpretable and Well-Documented Models

The Basel II Accord introduced risk-sensitive capital requirements through the Internal Ratings-Based (IRB) approach, where banks must estimate key risk parameters such as Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).

To comply with these requirements, credit risk models must be transparent, auditable, and well-documented. Regulators and internal risk teams must be able to understand how risk predictions are generated and which variables drive them.

Interpretability is essential because credit decisions directly affect customers and financial stability. A model that cannot be explained may fail regulatory validation, resulting in higher capital requirements under standardized approaches or rejection for IRB approval.

Therefore, model selection must prioritize not only predictive performance but also explainability, stability, and full documentation of assumptions and feature behavior.

---

## 2. Necessity of a Proxy Variable and Associated Risks

In this project, there is no direct default label or repayment history available in the dataset. To enable supervised learning, a proxy target variable must be constructed.

In this case, customer behavior from eCommerce transactions is summarized using Recency, Frequency, and Monetary (RFM) features. These behavioral signals are used to segment customers into high-risk and low-risk groups, serving as an approximation of creditworthiness.

While this enables model training, it introduces several risks:

- **Label noise**: The proxy does not represent actual loan default behavior.
- **Behavioral bias**: The model may learn spending patterns instead of true credit risk.
- **Misalignment risk**: Proxy-defined risk may not match real-world repayment outcomes.
- **Stability risk**: Relationships between behavior and risk may change over time.

As a result, proxy-based models must be carefully validated and continuously monitored to ensure they remain meaningful for credit decision-making.

---

## 3. Trade-offs Between Interpretable and High-Performance Models

Credit risk modeling requires balancing interpretability with predictive performance in a regulated environment.

### Logistic Regression with Weight of Evidence (WoE)

This approach is widely used in traditional credit scoring due to its simplicity and regulatory acceptance.

**Advantages:**
- Highly interpretable and easy to explain
- Clear relationship between variables and risk
- Strong regulatory acceptance under Basel II
- Easy to validate, monitor, and document

**Limitations:**
- Limited ability to capture nonlinear relationships
- May underperform on complex datasets

---

### Gradient Boosting Models (e.g., XGBoost, LightGBM)

These models are widely used in modern credit risk systems due to their strong predictive performance.

**Advantages:**
- Captures nonlinear relationships and feature interactions
- Often achieves higher predictive accuracy
- Works well with large and complex datasets

**Limitations:**
- Less interpretable than traditional models
- Requires explainability tools (e.g., SHAP) for justification
- More complex validation and governance process

---

### Summary

In regulated financial contexts, model selection is not based solely on accuracy. Institutions must balance predictive performance with interpretability, regulatory compliance, and operational transparency.

For this reason, both interpretable and high-performance models are typically evaluated before selecting a final model for deployment.# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank's Buy-Now-Pay-Later (BNPL) service in partnership with an eCommerce platform.

Since the dataset does not contain a direct default label, customer transaction behavior will be used to construct a proxy target variable using RFM (Recency, Frequency, Monetary) analysis. Multiple machine learning models will be evaluated to estimate risk probability, generate credit scores, and support lending decisions.

The solution will include:

- Feature engineering pipeline
- Credit risk model development
- Experiment tracking with MLflow
- REST API deployment using FastAPI
- Containerization using Docker
- CI/CD automation with GitHub Actions
