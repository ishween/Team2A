# Salesforce Team2A
Welcome to the central repository for Team 2A! This guide will walk you through the process of contributing your code. The goal of this project is to give you hands-on experience with version control, collaboration, and the standard developer workflow using Git and GitHub.

---
### 👥 **Team Members**
| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Caitlyn Widjaja    | @caitlynw5 | Data exploration, visualization, model training, feature engineering, datavalidation           |
| Afifah Hadi   | @hadiafifah     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Sai Wong     | @cywlol  | Data preprocessing, feature engineering, data validation                 |
| Mariam Jammal     | @mjamm-inc      | Model selection, hyperparameter tuning, model training and optimization  |
| Anusri Nagarajan       | @anusrinagarajan    | Model evaluation, performance analysis, results interpretation           |
| Mya Barragan       | @myabarragan    | Model evaluation, performance analysis, results interpretation           |
---

## 🎯 **Project Highlights**

- Developed an end-to-end machine learning and natural language transformation pipleline using ElasticNet regression, Random Forest, Gradient Boosting classifiers, and sentence transformers with vector embeddings to address comprhensive CRM sales optimization across account health scoring, lead qualification, and opportunity win prediction, and natural language data retrieval.
- Achieved R² of 0.947 for account health prediction, PR-AUC of 0.997 for lead scoring, F1-score of 0.973 for opportunity win prediction, and semantic search capabilities over 8,800 opportunities, demonstrating production-ready performance across predictive and retrieval tasks for Salesforce CRM optimization and conversational analytics.
- Generated actionable insights to inform business decisions at sales operations, account management, and revenue teams, including account prioritization strategies, lead qualification workflows, deal forecasting that captured 99.7% of winning opportunities, and natural language query interface enabling stakeholders to access CRM insights without SQL or technical expertise at Salesforce.
- Implemented comprehensive data preprocessing with feature engineering (70+ features), synthetic target generation, stratified cross-validation, hyperparameter tuning, rigorous data leakage prevention, and sentence-to-embedding transformation with ChromaDB vector storage to address real-world CRM constraints including missing labels, severe class imbalance, temporal dependencies, and the need for non-technical stakeholder access to complex sales data.

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

  ```
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
  pip install sentence-transformers chromadb
  pip install jupyter notebook ipykernel

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
Execute the notebooks in order: account health, lead scoring, opportunity win, sentence transformer

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.

