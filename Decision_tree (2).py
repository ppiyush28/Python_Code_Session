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
clf_dt = DecisionTreeClassifier(criterion = "entropy",max_depth=7,min_samples_split = 5 ) # Define model
clf_dt.fit (X_train, y_train) # Fit model with your data
predictions = clf_dt.predict(X_test) # Preditions on test data
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


#Decision Tree
from sklearn import tree
#clf_dt = tree.DecisionTreeClassifier(max_depth=10) # Define model
with open(g_path + "\decisionTree1.txt", "w") as f:
    f = tree.export_graphviz(clf_dt, out_file=f)


from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test,pred)


#For Loop For parameter search
seed = [0,10,45]
n_tree = [10,50,1000]
acc = []; model_list = []
for i , j in zip(seed,n_tree):
    print (" i, j ==>", i, j)
    model = RandomForestClassifier(n_estimators=j, criterion = "entropy", max_features = 3,
                                   max_depth =5, min_samples_split =5 ,random_state = i)
    model.fit(X_train,y_train)
    model_list.append(model)
    predictions_RF = model.predict(X_test)   
    acc.append(accuracy_score(y_test,predictions_RF))
        


#grid_dc = GridSearchCV(clf_dt,X_train, y_train, cv = 3)


 
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(n_estimators=100 )
model.fit(X_train,y_train)

predictions = model.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

y_test


#clf_dt.score (X_test, y_test) #accuracy score for test

"""
shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)
def test_classifier(clf):
    scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std())) 
    
test_classifier(clf_dt)

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)


eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
test_classifier(eclf)
"""

#with open(g_path + "decisionTree.txt", "w") as f:
 #   f = tree.export_graphviz(clf_dt, out_file=f)

    
# paste the content of decisionTree.txt in http://webgraphviz.com/    
X.shape
X[:4]
processed_df.head()


# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(n_estimators= 100)
model_RF.fit(X_train,y_train)
pred= model_RF.predict(X_test)
print(accuracy_score(y_test,pred))



# Grid search in Random Forest
from sklearn.model_selection import GridSearchCV
param_RF = { "n_estimators"      : [10,50],
           "criterion"         : ["gini", "entropy"],
           "max_features"      : [4,5],
           "max_depth"         : [10,15],
           "min_samples_split" : [2,5] }

grid_RF = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_RF,cv = 3,scoring = "accuracy")
grid_RF.fit(X_train , y_train )
print(grid_RF.best_score_)
print(grid_RF.best_params_)

final_modelRF = grid_RF.best_estimator_
final_modelRF.fit(X_train,y_train)
pred = final_modelRF.predict(X_test)
print(accuracy_score(y_test,pred))






from sklearn.ensemble import AdaBoostClassifier  as ADA
model = ADA(base_estimator=final_modelRF, n_estimators=50, learning_rate=1.0)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test,pred))





#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred= lr.predict(X_test)
pred_prob = lr.predict_proba(X_test)
print(accuracy_score(y_test,pred))




