import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("credit_data.csv")

# These are the columns we will focus on, this will keep indices
features = credit_data[["income", "age", "loan"]]
# This column is the target
target = credit_data.default

# machine learning handles arrays not data-frames
# (need to get rid of indices and column names and turn into array)
X = np.array(features).reshape(-1,3)
y = np.array(target)

model = LogisticRegression()
# cv sets the number of folds, default is 3
# will return array of length of fold numbers and will have the accuracy of algorithm in each fold
predicted = cross_validate(model, X, y, cv = 5)

print(predicted['test_score'])