import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from sklearn.linear_model import LogisticRegression
import pymc3 as pm
from sklearn.model_selection import train_test_split

## 1 point
diab_df = pd.read_csv("diabetes.csv")  # Update with the correct filename
print(diab_df.head())
## ToDo: read the csv file into a dataframe and show the first 5 rows


## 3 points

## Assign the Outcome variable to y and the rest to X.
## USe LogisticRegression to fit the data and print out the intercept and the coefficients
# Assign features to X and target to y
X = diab_df.drop("Outcome", axis=1)
y = diab_df["Outcome"]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Print the intercept and coefficients
print("Intercept:", model.intercept_[0])
print("Coefficients:", model.coef_[0])

## 2 points

## Explain what the code is doing:
## What are the prior probabilities of the intercept and coefficients?
# Which likelihood distribution has been used?
## What does pm.invlogit(linreg) mean?
## What is map_est?

with pm.Model() as logreg_model:
    w0 = pm.Normal('w0', mu=0, sd=100)
    w1 = pm.Normal('w1', mu=0, sd=100)
    w2 = pm.Normal('w2', mu=0, sd=100)
    w3 = pm.Normal('w3', mu=0, sd=100)
    w4 = pm.Normal('w4', mu=0, sd=100)
    w5 = pm.Normal('w5', mu=0, sd=100)
    w6 = pm.Normal('w6', mu=0, sd=100)
    w7 = pm.Normal('w7', mu=0, sd=100)
    w8 = pm.Normal('w8', mu=0, sd=100)

    linreg = w0 * np.ones(diab_df.shape[0]) + w1 * diab_df.Pregnancies.values + w2 * diab_df.Glucose.values \
             + w3 * diab_df.BloodPressure.values + w4 * diab_df.SkinThickness.values + w5 * diab_df.Insulin.values + \
             w6 * diab_df.BMI.values + w7 * diab_df.DiabetesPedigreeFunction.values + w8 * diab_df.Age.values
    p_outcome = pm.invlogit(linreg)

    likelihood = pm.Bernoulli('likelihood', p_outcome, observed=diab_df.Outcome.values)

    map_est = pm.find_MAP()
    print(map_est)

'''1. In Bayesian Logistic Regression, you specify prior distributions for the intercept (w0) and the coefficients (w1 to w8).
    pm.Normal('w', mu=0, sd=100):
      Each coefficient is assigned a normal prior with a mean (mu) of 0 and a standard deviation (sd) of 100.
    The intercept (w0) also has the same prior distribution.
    All coefficients and intercepts have normal priors with mean = 0 and standard deviation = 100.
   2. The likelihood is modeled using a Bernoulli distribution.
      Bernoulli distribution: It is used for binary classification (Outcome is either 0 or 1).
      p_outcome: The predicted probabilities of the positive class (diabetes).
      observed=diab_df.Outcome.values: The true labels from the dataset.
   3. The pm.invlogit() function is the inverse logit function, also known as the sigmoid function.
      It transforms the linear combination of the features into a probability between 0 and 1.
      Formula:invlogit(𝑥)=1/(1+e^(−x))
      linreg: The linear combination of features with their respective weights (coefficients).
      p_outcome: The predicted probability of the positive class (diabetes) for each instance.
   4. pm.find_MAP() finds the Maximum A Posteriori (MAP) estimate.
      The MAP estimate is the most probable values of the parameters given the data and the priors.
      It is similar to Maximum Likelihood Estimation (MLE) but incorporates the priors.
'''
## 2 points

with logreg_model:
    step = pm.Metropolis()  # Metropolis-Hastings step
    trace = pm.sample(400, step=step, return_inferencedata=True)
## ToDo: draw 400 samples using pm.Metropolis() and assign to the variable trace

## Explain the output of the plot
az.plot_posterior(trace)
