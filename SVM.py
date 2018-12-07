# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:26:04 2018

@author: DH384961
"""



import numpy as np # libraries for array operations
import pandas as pd # for data handling
from sklearn import  preprocessing #data sampling,model and preprocessing 

path ="E:\Class\Data\Titanic_train.csv"
#g_path = "./"
titanic_df = pd.read_csv(path) # read Data

titanic_df.head() # print few data


""" Data exploration and processing"""
titanic_df['Survived'].mean()


titanic_df.groupby('Pclass').mean()

class_sex_grouping = titanic_df.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()       


group_by_age = pd.cut(titanic_df["Age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()     

titanic_df.count()

titanic_df = titanic_df.drop(['Cabin'], axis=1)   

 
titanic_df = titanic_df.dropna()    

titanic_df.count()

""" Data preprocessing function"""
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['PassengerId','Name','Ticket'],axis=1)
    return processed_df
    

processed_df = preprocess_titanic_df(titanic_df)


X = processed_df.drop(['Survived'], axis=1).values # Features dataset
y = processed_df['Survived'] # Target variable

#Train Test split
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state = 100)

from sklearn.metrics import accuracy_score # Evaluation Metric 
from sklearn.svm import SVC # import SVM Classifier
svc_default = SVC() # Default Model
svc_default.fit(X_train,y_train)
pred1= svc_default.predict(X_test)
print(accuracy_score(y_test,pred1))

svc_1 = SVC(C= 0.1,kernel = "linear") # Model with defined parameters
svc_1.fit(X_train,y_train)

pred2= svc_1.predict(X_test)
print(accuracy_score(y_test,pred2))


from sklearn.model_selection import cross_val_score as cv
cv_score = cv(svc_default, X_train, y_train, cv=3)# CV score for default model

cv2_score = cv(svc_1, X_train, y_train, cv=3)# CV score for model predefined parameters

# Grid Searchin search of best optimal SVM parameters
from sklearn.model_selection import GridSearchCV # Import grid search function

# define grid values for each SVM Parameter
params = [
              {'C': [1,2,3], 'kernel': ['linear']}, 
              {'C': [10,3], 'gamma': [0.001,0.2], 'kernel': ['rbf']}
         ]

# define  grid  search          
grid_svc = GridSearchCV(estimator=svc_default, param_grid=params,cv =3)
grid_svc.fit(X_train , y_train ) # perform Grid search on data set
grid_svc.best_params_  # Best Parameters amog the grid valued that is supllied

grid_svc.best_score_  # Best Corresponding score to the best grid parameter

grid_svc.best_estimator_ # Final best Model

final_model = grid_svc.best_estimator_  # Define Best model

final_model.fit(X_train , y_train)


pred_cv = final_model.predict(X_test)
print(accuracy_score(y_test,pred_cv))

print(grid_svc.best_score_)
print(grid_svc.best_params_)













#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred= lr.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))



