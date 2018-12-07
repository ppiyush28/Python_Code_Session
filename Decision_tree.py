# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# URL: http://www.webgraphviz.com/

import numpy as np # libraries for array operations
import pandas as pd # for data handling
from sklearn import  cross_validation, preprocessing #data sampling,model and preprocessing 

path ="E:\Class\Data\Titanic_train.csv"
g_path = "E:\Class"
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


X = processed_df.drop(['Survived'], axis=1) # Features dataset
y = processed_df['Survived'] # Target variable

#train test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=50)


 
#X_train = pd.DataFrame(X_train)
#Model Implementation
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(criterion = "gini",max_depth=7,min_samples_split = 5 ) # Define model
clf_dt.fit (X_train, y_train) # Fit model with your data
predictions = clf_dt.predict(X_test) # Preditions on test data
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


#Decision Tree
from sklearn import tree
#clf_dt = tree.DecisionTreeClassifier(max_depth=10) # Define model
with open(g_path + "\decisionTree1.txt", "w") as f:
    f = tree.export_graphviz(clf_dt, out_file=f)

