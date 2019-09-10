################ import libraries ###########################
import pandas as pd
from matplotlib import pyplot as plt
##load dataset 
import pandas as pd
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

X_train = raw_data.iloc[:60,1:-3]
y_train = raw_data.iloc[:60,-2]

X_test = raw_data.iloc[60:,1:-3]
y_test = raw_data.iloc[60:,-2]
################ baseline : classic linear regression  ################
# create regression linéar modele
lr = linear_model.LinearRegression()

# train the model on training set
lr.fit(X_train,y_train)

# get the nomr 2 error on training set as baseline 
baseline_error = np.mean((lr.predict(X_test) - y_test) ** 2)
print(baseline_error)

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
plt.plot(alphas, scores, '-ko')
plt.axhline(lassocv_score, color='b', ls='--')
plt.axvline(lassocv_alpha, color='b', ls='--')
plt.xlabel(r'$\alpha$')
plt.ylabel('Score')
plt.xscale('log')
sns.despine(offset=15)
