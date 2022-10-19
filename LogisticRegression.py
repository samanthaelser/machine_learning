import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

X = np.array(
    [[0], [0.6], [1.1], [1.5], [1.8], [2.5], [3], [3.1], [3.9], [4], [4.9], [5], [5.1],
     [3], [3.8], [4.4], [5.2], [5.5], [6.5], [6], [6.1], [6.9], [7], [7.9], [8], [8.1]])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# plt.plot(x1, y1, 'ro', color='blue')
# plt.plot(x2, y2, 'ro', color='red')
# error with redundant color keyowrd?^
plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'ro')
# plt.show()

model = LogisticRegression()
model.fit(X,y)

print("b0 is:", model.intercept_)
print("b1 is: ", model.coef_)

def logistic(classifier, x):
    return 1 / (1+ np.exp(-(model.intercept_ + model.coef_ * x)))

for i in range(1, 120):
    # plt.plot(i/ 10.0 - 2, logistic(model, i / 10.0), 'ro', color='green')
    plt.plot(i / 10.0 - 2, logistic(model, i / 10.0), 'go')


plt.axis([-2, 10, -0.5, 2])
plt.show()

pred = model.predict([[1]])
pred2 = model.predict_proba([[1]])
# gives array with probability value belongs to first class
# and prob value belongs to second class

print("Prediction: ", pred)
print("Prediction Probability: ", pred2)

