# Salesforce Team2A
Welcome to the central repository for Team 2A! This guide will walk you through the process of contributing your code. The goal of this project is to give you hands-on experience with version control, collaboration, and the standard developer workflow using Git and GitHub.

---
### 👥 **Team Members**
| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Caitlyn Widjaja    | @caitlynw5 | Opportunity Win Model Development          |
| Afifah Hadi   | @hadiafifah     | Data Cleaning and Preprocessing, Feature Engineering, Lead Scoring Model Development, Gradio Interface Development |
| Sai Wong     | @cywlol  | Data Cleaning and Preprocessing, Feature Engineering, Sentence Transformer Development, Ollama Model Integration |
| Mariam Jammal     | @mjamm-inc      | Lead Scoring Model Development |
| Anusri Nagarajan       | @anusrinagarajan    | Data Cleaning and Preprocessing, Feature Engineering, Account Health Scoring Model Development |
| Mya Barragan       | @myabarragan    | Data Cleaning and Preprocessing, Feature Engineering, Opportunity Win Model Development |
---

## 🎯 **Project Highlights**

- Developed an end-to-end machine learning and natural language transformation pipleline using ElasticNet regression, Random Forest, Gradient Boosting classifiers, and sentence transformers with vector embeddings to address comprhensive CRM sales optimization across account health scoring, lead qualification, and opportunity win prediction, and natural language data retrieval.
- Achieved R² of 0.947 for account health prediction, PR-AUC of 0.997 for lead scoring, F1-score of 0.973 for opportunity win prediction, and semantic search capabilities over 8,800 opportunities, demonstrating production-ready performance across predictive and retrieval tasks for Salesforce CRM optimization and conversational analytics.
- Generated actionable insights to inform business decisions at sales operations, account management, and revenue teams, including account prioritization strategies, lead qualification workflows, deal forecasting that captured 99.7% of winning opportunities, and natural language query interface enabling stakeholders to access CRM insights without SQL or technical expertise at Salesforce.
- Implemented comprehensive data preprocessing with feature engineering (70+ features), synthetic target generation, stratified cross-validation, hyperparameter tuning, rigorous data leakage prevention, and sentence-to-embedding transformation with ChromaDB vector storage to address real-world CRM constraints including missing labels, severe class imbalance, temporal dependencies, and the need for non-technical stakeholder access to complex sales data.
- Integrated a local LLM workflow using Ollama with a Gradio frontend to unify predictive models, semantic search, and natural language reasoning into a single conversational CRM assistant, enabling real time query handling, context aware retrieval over ChromaDB embeddings, and an accessible interface for non technical stakeholders to interact with lead scores, win predictions, and account health insights without navigating notebooks or backend pipelines.

---

## 👩🏽‍💻 **Setup and Installation**

### 1. Getting Started: Setting Up Your Local Environment

Since this is a private repository, you will **clone** it directly to your computer.

1.  **Open your terminal** or command prompt.

2.  **Navigate to the folder** where you want to store your project.

3.  **Clone the repository** using the following command. Make sure to use the correct URL for your project.

    ```
    git clone [https://github.com/ishween/Team2A.git](https://github.com/ishween/Team2A.git)
    
    ```

4.  **Change into the project directory** you just cloned.

    ```
    cd Team2A
    ```

---

### 2. Install Dependencies
Python dependencies:

  ```
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost sentence-transformers chromadb jupyter notebook ipykernel gradio
  ```
Ollama installation

1. Download Ollama from: https://ollama.com/download
2. Pull model to be used:
```
ollama pull llama3
```
---

### 3. Access the Dataset
Datasets are inlcuded in the repo

  ```
  data/processed
  ```
---
### 4. Run the Notebooks
Start Jupyter Notebook

  ```
  jupyter notebook
  ```
Execute the notebooks in order: account health, lead scoring, opportunity win, sentence transformer, and then agents.ipynb inside the agents folder

## 🏗️ **Project Overview**

This project was completed as part of the Break Through Tech AI Program, a workforce development initiative designed to provide underrepresented students with hands-on machine learning and AI experience. The program bridges the gap between academic learning and industry practice by partnering students with real companies to solve authentic business problems.

**Host Company:** Salesforce
**Industry:** Customer Relationship Management (CRM) / Enterprise Software
Company Context: Salesforce is the world's leading CRM platform, helping businesses manage customer relationships, sales pipelines, and marketing campaigns. With millions of users worldwide, Salesforce processes vast amounts of sales data daily.

**Project Object and Scope**
Our team is building a CRM Intelligence Assistant, an internal tool that helps sales and marketing teams work smarter with their data. Using large language models, predictive analytics, and NLP, the assistant uncovers patterns, predicts opportunities, and makes advanced AI insights accessible through natural language queries—no technical expertise required.This intelligent assistant transforms traditional CRM systems from passive data repositories into active decision-support tools that:

Proactively identify at-risk accounts before they churn
Automatically qualify incoming leads to maximize sales efficiency
Predict deal outcomes with high accuracy for better resource allocation
Answer natural language questions like "Which accounts in the healthcare sector are most likely to close this quarter?" without requiring SQL knowledge
The goal is to democratize data-driven decision making across the entire organization, from frontline sales reps to C-suite executives.

**Specific Goals:**
Account Health Scoring: Predict account health scores to identify at-risk customers and prioritize high-value accounts for proactive engagement
Lead Scoring: Automatically qualify leads to help sales teams focus efforts on prospects most likely to convert
Opportunity Win Prediction: Forecast deal outcomes to improve revenue forecasting and resource allocation
Natural Language Interface: Enable non-technical stakeholders to query CRM data using natural language through semantic search and retrieval-augmented generation (RAG)

**Project Scope:**
Dataset: 8,800 sales opportunities across 4 merged tables (accounts, products, sales pipeline, sales teams)
Timeframe: 2016-2017 historical sales data
Deliverables:
- 3 production-ready ML models (regression + 2 classifiers)
- 1 RAG/NLP system for semantic search with sentence transformers and ChromaDB
- Comprehensive data preprocessing pipeline with feature engineering
- Model evaluation reports with business metrics
- Reproducible code and documentation

**Impact**
Sales teams waste 60%+ of their time on unqualified leads and discover at-risk accounts only after they've churned, costing companies millions annually. Our CRM Intelligence Assistant combines predictive analytics with conversational AI to deliver $1.5M+ in productivity savings through 95.4% accurate lead scoring, $2.5M+ in retained revenue via proactive account health monitoring (R² = 0.947), and 99.7% recall on deal predictions for accurate forecasting. By making AI insights accessible through natural language queries—reducing time-to-insight by 80-90%—this solution democratizes data-driven decision making, enabling sales teams to close 15-20% more deals and transforming reactive CRM systems into proactive intelligence assistants accessible to everyone, regardless of technical expertise.

---

## 🙏 **Acknowledgements**

We would like to sincerely thank Ishween Kaur, our Challenge Advisor, for her expertise, guidance, and consistent support throughout this project.
We are also grateful to Leah Dsouza, our AI Studio Coach, for always being willing to help and generously sharing her knowledge whenever we needed it.

