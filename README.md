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

- Developed a natural language transformation pipleline using ElasticNet regression, Random Forest, and Gradient Boosting classifiers to address comprhensive CRM sales optimization across account health scoring, lead qualification, and opportunity win prediction.
- Achieved exceptional model performance across all objectives:
  - Account Health Model -- (R² = 0.947, MAE = 4.89)
  - Lead Scoring Model -- (PR-AUC = 0.997, ROC-AUC = 0.952, 95.4% accuracy)
  - Opportunity Win Model -- (PR-AUC = 0.997, ROC-AUC = 0.952, 95.4% accuracy)
  This demonstrates production ready predictive accuracy that exceeded project targets    R² ≥ 0.70)for Salesforce's data-driven sales optimization.
- Generated actionable insights to inform business decisions for sales teams, account managers, and revenue operations at Salesforce .
- Implemented advanced ML engineering practices including synthetic target generation with weighted feature construction, extensive feature engineering (70+ variables from 4 source tables), one-hot and label encoding for categorical variables, log transformations for skewed distributions, temporal feature extraction, stratified K-fold cross-validation (5 folds), RandomizedSearchCV + GridSearchCV hyperparameter optimization (50 iterations), and comprehensive data leakage prevention to address industry-standard constraints in CRM analytics including unlabeled data, severe class imbalance (6:1 and 3:1 ratios), multicollinearity, and temporal data splitting.

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

# 2. Making Your Contributions: The Branching Strategy

For this project, you will create a new branch for your work. This keeps your changes separate from the main project until they are ready to be reviewed.

1.  **Create a new branch** using a descriptive name that includes your name. This is a crucial step for collaboration.

    ```
    git checkout -b your-name-2A
    
    ```

2.  **Make your code changes.** Do your work, add new files, and modify existing ones as needed.

3.  **Add and commit your changes.** Once you've completed a logical piece of work, save your changes to your branch. Be sure to write a clear and concise commit message.

    ```
    git add .
    git commit -m "Add my solution to Problem 1"
    
    ```

---

# 3. Submitting Your Work: The Pull Request

Once you are finished with your work, you will push your branch to the central repository and open a Pull Request (PR). A PR is how you propose your changes to be merged into the main project.

1.  **Push your branch** to the central repository.

    ```
    git push origin your-name-2A
    
    ```

2.  **Open a Pull Request.** Go to the GitHub repository page in your browser. GitHub will likely show a banner at the top asking if you want to create a new PR for the branch you just pushed. Click on that. If not, go to the "Pull requests" tab and click "New pull request."

3.  **Fill out the PR details.**

    * Make sure the base branch is `main` (the default) and the compare branch is `your-name-2A`.

    * Write a clear title and description. Explain what you've done and any challenges you faced. This helps the project maintainer understand your work.

4.  **Create the pull request.** Your PR is now open and serves as your final submission!

---

# What Happens Next?

The project maintainer will receive a notification about your pull request. They will review your code on your individual branch. Your work will not be merged into the `main` branch. Your `individual branch` will serve as your final submission.

If you have any questions, please reach out to the project maintainer. Good luck!
