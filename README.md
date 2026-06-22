# IBM HR Analytics — Employee Attrition Analysis

End-to-end data science project analyzing employee attrition with predictive 
modeling, risk scoring, and cost-benefit interventions.

🔗 **[Live Dashboard](https://your-username-ibm-hr-dashboard.streamlit.app)** ← ganti dengan link setelah deploy

## Key Findings
- **Overtime** is the strongest attrition driver (30.5% vs 10.4%)
- **Early-career employees** (0-2 years) show 44% attrition rate
- **Overtime + Low Salary** combination = 58.5% attrition (departure trigger)
- **Stock options** reduce attrition risk by 32% per level

## Business Impact
- **Investment:** $520K–$750K + equity
- **Savings:** $4M–$7M+ in avoided replacement costs
- **ROI:** 5–10x return
- **Target:** Attrition from 16.1% → below 10%

## Dashboard Preview
![Executive Dashboard](assets/dashboard_preview.png)

Two-page interactive dashboard (Streamlit):
- **Executive View** — KPIs, driver breakdown, risk distribution, ROI
- **HR Action View** — filterable risk table, department breakdown, employee-level scores

## Project Structure
| Section | Description |
|---|---|
| 0 | Executive Summary |
| 1-3 | Project Overview, Business Context, Business Questions |
| 4-5 | Data Loading, Cleaning & Preprocessing |
| 6 | Exploratory Data Analysis (Chi-Square validation) |
| 7 | Feature Engineering |
| 8 | Modeling (Logistic Regression + Random Forest + ROC/PR) |
| 9 | Model Interpretation (Odds Ratios, Feature Importance) |
| 10-11 | Risk Scoring & Decision Framework |
| 12 | Business Intervention (4 prioritized, cost-justified + cost-benefit methodology) |
| 13-14 | Experiment Design & Monitoring Framework |
| 15 | Conclusion |
| 16 | Limitations & Future Work |

## Tools & Methods
- Python (pandas, scikit-learn, matplotlib, seaborn, scipy)
- Streamlit + Plotly (interactive dashboard)
- Logistic Regression (selected for interpretability) + Random Forest (benchmark)
- Chi-Square test, ROC-AUC, PR-AUC, Odds Ratios
- SHRM industry benchmarks for cost estimation

## Files
| File | Description |
|---|---|
| `IBM_HR_Analytics.ipynb` | Complete 16-section analysis notebook |
| `dashboard.py` | Interactive Streamlit dashboard (2-page) |
| `IBM_HR_CEO_Deck.pptx` | 14-slide executive presentation |
| `requirements.txt` | Python dependencies |

## Data Source
IBM HR Analytics Employee Attrition & Performance  
Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
1,470 employees, 35 features, cross-sectional

## Quick Start
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Author
**Zikra Aditia** — People Analytics & Data Science  
[LinkedIn](https://linkedin.com/in/your-url)
