import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris_X, iris_Y = datasets.load_iris(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

#Predict
y_pred = knn_classifier.predict(X_test)
accuracy_score(y_test,y_pred)