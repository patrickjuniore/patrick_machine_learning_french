################ import libraries ###########################
##load dataset 
import pandas as pd
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
## 
from sklearn import linear_model
import numpy as np
##vizualization
from matplotlib import pyplot as plt
import seaborn as sns

################ load dataset ###########################
#raw_data = pd.read_csv('prostate.data.txt', delimiter='\t')
raw_data = pd.read_csv(r"C:\Users\p_michel-ext\patrick_projets\perso\data\TP_1_prostate_dataset.txt", delimiter='\t')

#LassoCV

X = raw_data.iloc[:60,1:-3]
Y = raw_data.iloc[:60,-2]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
################ baseline : classic linear regression  ################
# Fitting Simple Linear Regression to the Training set

lr = linear_model.LinearRegression() # create regression linéar modele
lr.fit(X_train,y_train) # train the model on training set

# get the nomr 2 error on training set as baseline 
baseline_error = np.mean((lr.predict(X_test) - y_test) ** 2)
print(baseline_error)
#2.8641499657014458

################ apply Lasso ##################################
n_alphas = 300
alphas = np.logspace(-5, 1, n_alphas)
scores = np.empty_like(alphas)

lasso = linear_model.Lasso(fit_intercept=False)

coefs = []
errors = []
for i,a in enumerate(alphas):
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    scores[i] = lasso.score(X_test, y_test)
    coefs.append(lasso.coef_)
    errors.append([baseline_error, np.mean((lasso.predict(X_test) - y_test) ** 2)])

lassocv = linear_model.LassoCV()
lassocv.fit(X_train, y=y_train)
lassocv_score = lassocv.score(X_train, y=y_train)
lassocv_alpha = lassocv.alpha_
print('CV', lassocv.coef_)

################ vizualization ##################################
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.axis('tight')
plt.show()

ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.axis('tight')
plt.show()

plt.plot(alphas, scores, '-ko')
plt.axhline(lassocv_score, color='b', ls='--')
plt.axvline(lassocv_alpha, color='b', ls='--')
plt.xlabel(r'$\alpha$')
plt.ylabel('Score')
plt.xscale('log')
sns.despine(offset=15)
