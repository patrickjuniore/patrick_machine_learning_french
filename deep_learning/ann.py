# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:06:11 2019

@author: User
"""

################################### Importing libraries###################################
# Import data
import pandas as pd
# Pre-processing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
# Evaluate
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#https://github.com/scikit-learn/scikit-learn/issues/10054
#Turns out, it was n_jobs=-1 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# Tune
from sklearn.model_selection import GridSearchCV

################################### Importing data###################################
dataset = pd.read_csv(r"C:\Users\User\desk\machine_learning\deeplearning-master\deeplearning-master\Part 1 - Artificial_Neural_Networks\data\Churn_Modelling.csv")
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

################################### Pre-processing ###################################
#https://tomaugspurger.github.io/sklearn-dask-tabular.html
categorical_columns = ['Geography','Gender']
numerical_columns = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
#categorical_encoder = OneHotEncoder(sparse=False)
categorical_encoder = OneHotEncoder()


preprocess = make_column_transformer(
    (categorical_columns, OneHotEncoder()), #transform an entire format category into several columns in binary format.
    (numerical_columns, StandardScaler()),	#scaling the numeric values to avoid aberrations and increase performance.
    remainder='passthrough'
)

X = preprocess.fit_transform(X)

# Split in train/test
y = y.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

################################### Build the  ANN###################################

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

# classifier.add(Dropout(0.10)) #dropout here (start wth 10%)

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# classifier.add(Dropout(0.10)) #dropout here (start wth 10%)

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

################################### Run the  model: Fitting the ANN to the Training set ###################################
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
################################### Making new predictions ###################################
#### Predicting the Test set results
y_pred = classifier.predict(X_test)
#### predictions should be 1 or 0 to put the client into one category (binary catgory --> 0 or 1, into the category or not)
y_pred = (y_pred > 0.5)				  # require to build the confusion's matrix: the threshold is set at 0.5
cm = confusion_matrix(y_test, y_pred) # Making the Confusion Matrix
#### Example:Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
new_prediction = classifier.predict(Xnew)
# predictions should be 1 or 0 to put the client into one category (binary catgory --> 0 or 1, into the category or not)
# require to build the confusion's matrix: the threshold is set at 0.5
new_prediction = (new_prediction > 0.5)

################################### Evaluate the model  ###################################
## function to build a classifier
def build_classifier(optimizer='adam'):
    ######same as the previous step 'Build the  ANN'
	classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

##### cross-validation: divide the training set into 10 part then test one after another while the 9 other are used to train the model
## build a classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
## use the classifier build previously into cross-validation to calculate the accuracies for the 10 parts.
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                             cv = 10#
							 #, n_jobs = -1 #nbr de CPUs use to made the calculs | -1 mean use all the CPUs
                             )
mean = accuracies.mean()	 # mean for the 10 parts
variance = accuracies.std()	 # standard deviation for the 10 parts

################################### Tune the model (find the best hyperparameters)   ###################################
#GridSearchCV need  to know what optimize,here CLASSIFIER
classifier = KerasClassifier(build_fn = build_classifier ''',batch_size = 10, epochs = 100 (it's the parameers we want optimize)''')
#GridSearchCV need  a dictionnary of PARAMETERS to optimize them
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
#GridSearchCV look the best parameters/accuracy   based  CLASSIFIER and PARAMETERS
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_	#the best combinaison batch_size,epochs,optimizer
best_accuracy = grid_search.best_score_		#the best accuracy found
