# numerical-linear-regression
Linear regression from scratch in Python with focus on numerical stability (QR/SVD)

The goal is to compute a global regression vector and use it for predictions on new input data.

## Setup
- Create a virtual environment
- Install dependencies via pip install -r requirements.txt 


## Problem statement

The goal of this project is to estimate a global regression vector β
for a linear model by solving a least squares problem.

Given a data matrix X and target values y, the regression parameters
are obtained by minimizing

||Xβ − y||².

The estimated parameter vector is then used to generate predictions
for new input data.


## Scope

- The project focuses on linear regression only.
- Synthetic data is used to allow controlled numerical experiments.
- The regression problem is solved using numerically stable methods
  (QR decomposition, SVD).
- No machine learning frameworks such as scikit-learn are used.
- The focus is on understanding numerical stability, not on achieving
  maximum predictive accuracy.


## Project goals

The project is considered complete when:

- Synthetic regression data can be generated reproducibly.
- A global regression vector β can be computed using QR decomposition/SVD.
- The model can be used to generate predictions for new input data.
- The numerical behavior of the solution can be analyzed and visualized.


## Data generation

Synthetic data is generated to allow controlled numerical experiments.
The target values are constructed according to

y = Xβ_true + ε,

where β_true is a fixed, known parameter vector and ε denotes additive
Gaussian noise.

Two types of datasets are generated:
- a well-conditioned design matrix with independent features
- an ill-conditioned design matrix with strong collinearity

This setup makes it possible to study the numerical behavior and stability
of different solution methods under controlled conditions.

## Experiments and Evaluation

The generated data is split into training and test sets using an 80/20 split.
Model performance is evaluated on the test set using the root mean squared error (RMSE).
In addition, the estimated regression coefficients are compared to the true parameter
vector to assess numerical stability.

Two numerical solvers are compared:
- QR decomposition
- Singular Value Decomposition (SVD)

For well-conditioned design matrices, both solvers yield identical results, as expected.
For ill-conditioned problems, truncated SVD (TSVD) is used by discarding small singular
values below a specified tolerance. This acts as a form of numerical regularization,
leading to more stable parameter estimates and, in some cases, improved generalization
performance on the test set.