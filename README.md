# README

## Skills demonstrated

**Econometrics:**
- Logistic regression
- Marginal effects

**Machine learning:**
- Random forest
- Hyperparameter tuning

**Evaluation:**
- Credit risk modelling
- Feature engineering
- Classification metrics

**Python:**
- Pandas
- scikit-learn
- statsmodels

## Project Overview

The project investigates loan approval prediction by comparing the traditional econometric logistic regression and the ML random forest model, using data from a publicly available Kaggle dataset. Correctly predicting loan approval is important for banks in order to reduce the risk of default and therefore financial loss. The objective is to assess the trade-off between interpretability and predictive accuracy and evaluate which model banks may prefer depending on their objectives. The performance of each model is evaluated across multiple metrics including a classification report which defines the accuracy, precision, recall and F1 score, a confusion matrix, AUC and ROC, and additionally feature importance. A full research report including a literature review, reproducible Python notebook, and supporting documentation are included in this repository.

## Research Question

The report assesses whether machine learning can outperform traditional econometric methods in loan approval. Additionally, it aims to evaluate which modelling approach can offer the best balance between interpretability and predictive accuracy when granting loans to borrowers.

## Dataset

| Variable | Type | Model | Description |
|---|---|---|---|
| `loan_id` | Int | – | Applicant number |
| `no_of_dependents` | Int | X1i | Number of dependents of the applicant |
| `education` | Categorical | X2i | 'Graduate' or 'Not graduate' |
| `self_employed` | Categorical | X3i | Employment status of the applicant: 'Yes' or 'No' |
| `income_annum` | Int | X4i | Annual income of the applicant |
| `loan_amount` | Int | X5i | Loan amount |
| `loan_term` | Int | X6i | Duration of the loan in years |
| `cibil_score` | Int | X7i | Credit score of the applicant |
| `residential_assets_value` | Int | X8i | Value of residential assets of applicant |
| `commercial_assets_value` | Int | X9i | Value of commercial assets of applicant |
| `luxury_assets_value` | Int | X10i | Assets of luxury value owned by applicant |
| `bank_assets_value` | Int | X11i | Assets of applicant in bank |
| `loan_status` | Categorical | y | Binary outcome variable of loan approval or rejection: 'Approved' or 'Rejected' |

The dataset is publicly available on Kaggle and titled **Loan-Approval-Prediction-Dataset**: [kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset?select=loan_approval_dataset.csv)

It has 4,269 observations corresponding to the number of loan applicants and 12 predictor variables. The identifier variable `loan_id` was removed prior to training the model as it contains no predictive information. The 13th variable is the target variable defined as `loan_status`, which depicts whether the applicant has been 'approved' or 'rejected' for a loan. The data is split into an 80% training group and 20% test group before training begins.

## Repository Structure

| File | Description |
|---|---|
| `README.md` | Project overview, setup instructions, methodology and key findings |
| `Loan_approval_report.pdf` | Full research report with literature review, methodology, results and discussion |
| `loan_approval_RF.ipynb` | Random Forest implementation and analysis |
| `loan_approval_logistic.ipynb` | Logistic Regression implementation and analysis |
| `Requirements.txt` | Required Python packages |
| `LICENSE` | MIT License |
| `.gitignore` | Specifies files ignored by Git |

## Methodology

### Data Cleaning

The variable `loan_id` was dropped from the dataset as it doesn't act as a regressor in the model. There are no missing values that need to be accounted for, however there are 3 categorical variables that need to be encoded as binary variables:

| Variable | 0 | 1 |
|---|---|---|
| Education | Graduate | Not Graduate |
| Self_Employed | No | Yes |
| Loan_Status | Approved | Rejected |

### EDA

There is an uneven split of approvals and rejections in a 62:38 split (2,656 approvals and 1,613 rejections). This has been accounted for in the code to focus on the minority group, so the model doesn't predict more approvals just because it has seen more of them.

The correlation heatmap shows how input variables move with each other. Regressors including bank assets, commercial assets, residential assets, income per annum and loan amount appear to be correlated with most other regressors. This suggests there is multicollinearity in the logistic regression, therefore leading to inflated standard errors despite estimates remaining unbiased.

The VIF analysis suggests there is severe multicollinearity for regressors income per annum, luxury assets, loan amount and bank asset value, and therefore standard errors in the logistic regression will be inflated for these. Therefore, although marginal effects remain valid, the precise magnitude and statistical significance of individual marginal effects for these high-VIF variables are unreliable — variables that appear insignificant may have real effects the model can't detect.

However, likelihood ratio tests suggest these variables jointly contribute a meaningful fit despite multicollinearity.

### Logistic Regression

The logistic regression is used as the benchmark econometric model, as it is the traditional approach and has been widely applied in credit scoring. Model coefficients are estimated with maximum likelihood estimation and interpreted via marginal effects.

### Random Forest

The random forest is a machine learning (ML) model comprising a combination of decision trees trained in parallel. Machine learning models have increasingly been applied to credit scoring. The random forest undergoes hyperparameter tuning, which increased predictive accuracy.

### Evaluation

The test set was generated with a random seed of 1, providing a support of 521 approvals and 333 rejections. The results from both models were evaluated using:

- Classification report (accuracy, precision, recall and F1-score)
- Confusion matrix
- ROC curve and AUC
- Feature importance

All measures for the random forest show results before and after hyperparameter tuning.

## Visualisations

The notebook contains:

**Class distribution:**

<img width="207" height="96" alt="class distribution" src="https://github.com/user-attachments/assets/34ae3cae-1770-4168-a884-ba1b7c786b93" />

**Summary statistics:**

<img width="752" height="160" alt="SS1" src="https://github.com/user-attachments/assets/438cf80a-c437-4085-a3ae-f457ecc78e1f" />

<img width="473" height="336" alt="SS2" src="https://github.com/user-attachments/assets/d995d06e-0c86-42e0-b62f-369a2b168bb2" />

**Correlation heatmap:**

<img width="752" height="626" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/8fed3511-6146-4b56-bdd8-737cda431b4e" />

**ROC curves:**

*Logistic regression:*

<img width="590" height="457" alt="ROC_logistic" src="https://github.com/user-attachments/assets/095b6f0d-8de5-4db4-8ca3-da509b11d45d" />

*Tuned random forest:*

<img width="565" height="461" alt="ROC_tuned" src="https://github.com/user-attachments/assets/fd364f08-94b5-413a-96f9-7df4af1ee227" />

**Confusion matrices:**

*Logistic regression:*

<img width="590" height="450" alt="confusion_logistic" src="https://github.com/user-attachments/assets/7e8f3311-9145-408d-84e2-0765c4918c7a" />

*Untuned random forest:*

<img width="585" height="438" alt="confusion_untuned" src="https://github.com/user-attachments/assets/5db0dd3b-4539-4bc8-9d2e-1817b7a1416e" />

*Tuned random forest:*

<img width="580" height="437" alt="confusion_tuned" src="https://github.com/user-attachments/assets/bcf60445-f18b-447a-abbd-9bb59c26b6fa" />

**Feature importance:**

*Logistic regression:*

<img width="695" height="514" alt="feature_importance_logistic" src="https://github.com/user-attachments/assets/4f78481c-932f-4a42-9830-a3db4fc74177" />

*Random forest:*

<img width="638" height="476" alt="feature_importance_RF" src="https://github.com/user-attachments/assets/47c80fbf-cf24-4753-a65d-a3cb6cba0525" />

**Marginal effects plots:**

*Average marginal effects:*

<img width="737" height="352" alt="average_marginal_effects" src="https://github.com/user-attachments/assets/38c92c55-dab4-40ce-a82c-a098d9339a15" />

*Marginal effects at the mean:*

<img width="731" height="352" alt="ME_at_the_mean" src="https://github.com/user-attachments/assets/81c13ce2-ac52-4917-9943-515fc7f99b2b" />

## Results

| Metric | Logistic Regression | Random Forest – untuned | Random Forest – tuned |
|---|---|---|---|
| Approval precision | 0.94 | 1.00 | 1.00 |
| Rejection precision | 0.89 | 0.89 | 0.91 |
| Approval recall | 0.93 | 0.92 | 0.94 |
| Rejection recall | 0.91 | 0.99 | 0.99 |
| F1 score (approvals) | 0.94 | 0.96 | 0.97 |
| F1 score (rejections) | 0.90 | 0.94 | 0.95 |
| Overall accuracy | 0.92 | 0.95 | 0.96 |
| AUC | 0.965 | 0.997 | 0.998 |

The random forest outperforms the logistic regression in predictive accuracy across all quantitative measures shown in the table above.

## Key Findings

- The random forest model achieves the highest predictive accuracy
- The logistic regression remained competitive
- The cibil score feature is deemed to be the most important feature by both models
- The random forest model finds complex relationships between variables which the logistic regression classifies as linear
- The logistic regression outperforms from an interpretability perspective
- The logistic regression can also test variable significance
- The overall model choice depends on the objectives of banks and whether prediction or interpretability is prioritised

## Limitations

The data is unlikely to be fully representative — ideally it needs to reflect people likely to apply for credit in the future. Additionally, the data should incorporate different types of repayment behaviour.

The data can suffer from selection bias due to population drift, where populations evolve over time, changing the distributions.

The proportion of goods and bads must be considered — whether this is an equal split or instead reflects the odds in the population. Additionally, the sample size of goods and bads must also be considered.

Finally, results may not generalise to all banks, and it depends where the data was collected from. As Kaggle provides no descriptions, this is difficult to evaluate.

## How to Run

1. Clone the repository
2. Install the required Python packages
3. Open the notebooks in the recommended order:
   1. `loan_approval_logistic.ipynb`
   2. `loan_approval_RF.ipynb`
4. Run all the cells from top to bottom
5. Outputs, figures, and evaluation metrics will all be generated automatically

## Requirements

- Python 3.12
- pandas
- matplotlib
- scikit-learn
- scipy
- statsmodels
- kagglehub
- seaborn
