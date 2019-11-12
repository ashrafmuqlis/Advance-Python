#This data sets consists of 3 different types of irisesâ€™ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
#import sklearn as sk
from sklearn import datasets
#Load Data
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X
Y
# Split and Randomize Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
X_train
X_test


#Step 2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf=LogisticRegression()

#Step 3
clf.fit(X_train,Y_train)

#Step 4
pred = clf.predict(X_test)

#Step 5 
print ("Logistic Regression accuracy score : ", accuracy_score(Y_test,pred))
    
#VISUALIZATION
#conda install -c districtdatalabs yellowbrick
from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(clf, classes=[0,1,2])
visualizer.fit(X_train, Y_train) # Fit the training data to the visualizer
visualizer.score(X_test, Y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data
