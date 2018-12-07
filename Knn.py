# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:06:56 2017

@author: DH384961
"""

 
from sklearn import neighbors, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data # we only take the first two features. 
Y = iris.target

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.25, random_state=50)

k =3
model = neighbors.KNeighborsClassifier(k)
model.fit(X_train,y_train)
pred = model.predict(X_test)
sum(pred == y_test)/len(y_test) 
accuracy_score(y_test,pred)


k=[1,2,3,4,5,6,7,8,9,10]
accuracies = []

index = 0
for i in k:
    knn=neighbors.KNeighborsClassifier(i)
    # we create an instance of Neighbours Classifier and fit the data.
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test,predictions))
    print(accuracies[index])
    index = index + 1
    
import matplotlib.pyplot as plt
    
plt.plot(k,accuracies,'p-')
plt.xlabel(" -------K ----")
plt.ylabel(" ====Accuracies ==")
plt.title(" Accuracies for Different k values of KNN")
plt.xticks(range(15))



final_model = neighbors.KNeighborsClassifier(4)
final_model.fit(X_train,y_train)
pred = final_model.predict()



