# Credit Risk Analysis and Financial Decision Support System

This project is an end-to-end machine learning solution developed to predict **credit default risk** and support credit approval decisions in the banking sector using the **UCI German Credit Dataset**.

The main focus is not only statistical predictive performance, but also the financial consequences of incorrect decisions. The project therefore incorporates a **Business Value** framework that assigns different costs and benefits to different types of credit decisions.

## Project Overview

The system analyzes customer risk profiles using data from the UCI German Credit Dataset.

Model selection and threshold optimization are designed around a cost-sensitive decision framework, with particular emphasis on reducing the financial impact of **False Negative (FN)** decisions, where a high-risk customer is incorrectly approved.

## Technical Architecture and Model

The project uses a **Random Forest** classifier selected for its robustness to noisy and heterogeneous tabular data.

### Model Parameters

* `max_depth: 4` — shallow trees to reduce overfitting and sensitivity to noise
* `min_samples_leaf: 11`
* `n_estimators: 136`

### Preprocessing Pipeline

The preprocessing pipeline automatically handles numerical and categorical variables using:

* `StandardScaler`
* `OneHotEncoder`

### Decision Strategy

A **Growth-Oriented** decision strategy was used, with a classification threshold of **0.518**.

## Performance Results

The following results were obtained on the test dataset:

| Metric             |   Result   | Description                                         |
| :----------------- | :--------: | :-------------------------------------------------- |
| **ROC-AUC**        | **0.7829** | Overall discrimination performance                  |
| **Bad Recall**     |  **76.7%** | Proportion of defaulting loans correctly identified |
| **Good Precision** |  **87.9%** | Reliability of predictions classified as good       |
| **Approval Rate**  |  **74.0%** | Proportion of applications approved                 |

## Business Value and Cost Analysis

The model was optimized using the following decision-cost framework:

| Decision                       | Business Value |
| :----------------------------- | -------------: |
| **Preventing a Bad Loan (TP)** |   +1,000 units |
| **Approving a Good Loan (TN)** |     +200 units |
| **Rejecting a Good Loan (FP)** |     -150 units |
| **Approving a Bad Loan (FN)**  |   -5,000 units |

This framework reflects the asymmetric financial consequences of credit decisions rather than treating all classification errors as equally costly.

## Fairness Analysis

The project also includes a fairness analysis to examine whether model decisions differ across groups associated with selected attributes, including `Attribute9` and `Attribute17`.

The analysis produced a **35.0% Disparate Impact score** under the evaluated setup. This result highlights the importance of additional fairness monitoring and **human-in-the-loop review** when applying similar models in real-world credit decision systems.

Fairness metrics should be interpreted in the context of the dataset, selected attributes, threshold, and evaluation methodology rather than as standalone measures of overall model fairness.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
```

Run the analysis:

```bash
python credit_risk_analysis.py
```

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* ucimlrepo
* Random Forest
* Cost-sensitive decision analysis
* Fairness analysis

## Dataset

This project uses the **UCI German Credit Dataset**.

The dataset contains customer-level financial and demographic attributes used to classify credit applicants according to their credit risk.

## Conclusion

This project demonstrates how machine learning can be combined with **cost-sensitive decision analysis** and **fairness evaluation** in a financial risk setting.

Rather than optimizing solely for predictive accuracy, the project evaluates how different classification decisions can affect financial outcomes and examines potential differences in model behavior across groups.

The resulting system serves as a research and educational prototype for exploring machine learning, credit risk modeling, business-value optimization, and responsible AI considerations in financial decision-making.
