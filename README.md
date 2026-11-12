# Credit approval with machine learning

Predicting whether a credit card application is approved, and comparing several machine-learning approaches on the same data. The notebook covers data preparation, exploratory analysis, Bayesian networks, clustering, decision trees, random forests and neural networks.

Originally built as a portfolio for the *Data Mining and Machine Learning* course (F21DL) at Heriot-Watt University.

## Data

The [UCI Credit Approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval): 690 real credit card applications with 15 attributes and an approved/declined label. The original attribute names were anonymised. This project uses the commonly used descriptive names (Age, Debt, Income, CreditScore, YearsEmployed, PriorDefault and so on).

## What's in the notebook

1. **Data preparation and analysis:** handling missing values, encoding categorical attributes, summary statistics, correlations and scatter plots; Bayesian network modelling.
2. **Clustering:** k-means with the elbow method, and Gaussian mixture models (expectation maximisation), with a comparison of the two.
3. **Decision trees:** baseline tree, 10-fold cross-validation, the effect of tree depth and other parameters, and smaller training sets (50% and 80% held out for testing); then random forests.
4. **Neural networks:** multi-layer perceptrons with different layer sizes and settings.
5. **Research questions:** whether approval differs by ethnicity (a protected attribute), and how age, income, credit score and debt relate to approval.

## Results

Test accuracy on held-out data:

| Model | Accuracy |
|---|---|
| Decision tree, 10-fold cross-validation | **86%** (± 4%) |
| Decision tree, tuned parameters | 84–86% |
| Random forest, 10-fold cross-validation | 85% (± 5%) |
| Multi-layer perceptron (best setting) | 81% |

- **Employment status and credit score** have the strongest correlation with approval.
- **Deeper trees overfit.** Training accuracy rose to 93% while test accuracy fell to 78%.
- **Simple trees did as well as random forests** on this small dataset, and better than neural networks.

## Running it

```bash
pip install -r requirements.txt
cd F21DL
jupyter notebook credit_approval_ml.ipynb
```

The notebook reads `dataset/crxdata.csv`, relative to its own folder.

## Licence

The code is MIT licensed; see [LICENSE](LICENSE). The dataset is from the UCI Machine Learning Repository (contributed by J. R. Quinlan) and keeps its original terms.
