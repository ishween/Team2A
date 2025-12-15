# Smart CRM Helper  
### AI Assistant for Salesforce Sales Teams  
**Break Through Tech AI Studio | Host Company: Salesforce**

Smart CRM Helper is an intelligent conversational AI assistant built for Salesforce sales teams to reduce manual CRM work, surface high-value opportunities, and enable data-driven decision making through predictive analytics and natural language interaction.

---

## 👥 Team Members

| Name | GitHub Handle | Contribution |
|---|---|---|
| Afifah Hadi | @hadiafifah | Data cleaning and preprocessing, feature engineering, lead scoring model, Gradio interface |
| Sai Wong | @cywlol | Sentence transformers, ChromaDB integration, Ollama model integration |
| Caitlyn Widjaja | @caitlynw5 | Opportunity win prediction modeling |
| Mariam Jammal | @mjamm-inc | Lead scoring model development |
| Anusri Nagarajan | @anusrinagarajan | Account health scoring model |
| Mya Barragan | @myabarragan | Opportunity win prediction modeling |

---

## 🎯 Project Highlights

- Built three supervised machine learning models for lead scoring, opportunity win prediction, and account health scoring using Random Forest, Gradient Boosting, and ElasticNet.
- Achieved strong performance across tasks, including approximately 94 percent accuracy for lead scoring, approximately 96 percent accuracy and a 97.3 percent F1 score for opportunity win prediction, and an R² of 0.9466 for account health scoring.
- Designed a conversational AI workflow using sentence transformers, ChromaDB, Ollama, and Gradio to enable natural language CRM queries.
- Delivered measurable business value by reducing manual data search time and improving sales prioritization and retention outcomes.

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


## 🏗 Project Overview

This project was completed as part of the **Break Through Tech AI Studio**, a workforce development program that partners students with industry companies to solve real business problems using machine learning and AI.

**Host Company:** Salesforce  
**Industry:** Customer Relationship Management (CRM)

Sales teams spend a significant portion of their time manually searching CRM systems, prioritizing leads, and identifying at-risk accounts. This leads to lost opportunities, inefficient workflows, and reactive decision making.

Smart CRM Helper transforms Salesforce from a passive data repository into an active decision-support system by:

- Automatically scoring and ranking leads  
- Predicting opportunity win probabilities  
- Identifying at-risk and high-value accounts  
- Allowing users to ask natural language questions such as:  
  *Which healthcare accounts are likely to close this quarter?*

---

## 📊 Data Exploration

### Dataset Description

- Source: Salesforce CRM data  
- Size: Approximately 8,800 sales opportunities  
- Structure: Four merged tables including accounts, products, sales pipeline, and sales teams  
- Timeframe: 2016 to 2017 historical data  

### Data Preparation

Key preprocessing steps included:
- Handling missing values and duplicate records  
- Feature engineering for engagement, recency, and performance metrics  
- Standardization and encoding of categorical variables  
- Correlation analysis to prevent multicollinearity and data leakage  

### EDA Insights

- Revenue, recency, and engagement metrics showed the strongest correlation with account outcomes  
- Severe class imbalance required careful metric selection and validation strategies  
- Time-based splits were necessary to ensure temporal generalization  

Annotated visualizations, including feature importance plots and confusion matrices, are included within the notebooks.

---

## 🧠 Model Development

### Model Selection Rationale

- **Lead Scoring:** Random Forest was selected for its ability to capture non-linear interactions while maintaining strong precision.
- **Opportunity Win Prediction:** Gradient Boosting provided the best balance of accuracy, recall, and robustness.
- **Account Health Scoring:** ElasticNet handled correlated features and enabled interpretable weighting of business drivers.

### Training and Evaluation

- Stratified cross-validation for classification tasks  
- 80/20 train-test splits  
- Time-based validation for regression modeling  
- Metrics included Accuracy, Precision, Recall, F1 Score, ROC AUC, and R²  

---

## 📈 Results and Key Findings

| Model | Key Metrics |
|---|---|
| Lead Scoring | Approximately 94 percent accuracy |
| Opportunity Win Prediction | Approximately 96 percent accuracy, 97.3 percent F1 score, 99.7 percent of wins identified |
| Account Health Scoring | R² of 0.9466, time-based R² of 0.9324 |

These results exceeded baseline expectations and demonstrated readiness for real-world deployment.

---

## 🧩 Code Highlights

- `lead_scoring.ipynb`: Feature engineering and Random Forest training pipeline  
- `opportunity_win.ipynb`: Gradient Boosting model with leakage prevention  
- `account_health.ipynb`: Synthetic target construction and ElasticNet regression  
- `sentence_transformer.ipynb`: Embedding generation and ChromaDB vector storage  
- `agents.ipynb`: Agentic routing and Gradio conversational interface  

---

## 🧠 Discussion and Reflection

**What worked well:**
- Clear separation of predictive tasks  
- Strong alignment between business objectives and model outputs  
- Effective integration of NLP with traditional machine learning  

**Challenges:**
- Lack of ground truth labels for account health required synthetic target design  
- Class imbalance demanded careful evaluation strategy  
- Deployment constraints limited real-time integration  

---

## 🚀 Next Steps

- Add richer explanatory visualizations and outputs  
- Develop an interactive dashboard or web application  

---

## 📝 License

This project was completed for educational purposes as part of Break Through Tech AI Studio. No open-source license is currently applied.

---

## 🙏 Acknowledgements

We thank **Ishween Kaur**, Challenge Advisor, and **Leah Dsouza**, AI Studio Coach, for their guidance and support throughout this project.
