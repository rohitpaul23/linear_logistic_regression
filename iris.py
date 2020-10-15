import time

start = time.time()

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv('iris.csv', names=names)

'''
#Summarize the dataset
    #shape
print(dataset.shape)
    #head
print(dataset.head(10))
    #description
print(dataset.describe()) 
    #class distribution
print(dataset.groupby('class').size())
'''

'''
#Data Visualization
    #Univariate plots
        #box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
        #histogram
dataset.hist()
pyplot.show()

     #Multivariate plots
        #Scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
'''

#Evaluate Some Algo
    #Create a Validation Dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, y_train, y_validation = train_test_split(
                                                                X,
                                                                y,
                                                                test_size = 0.10,
                                                                random_state = 1)
    #Build Model
#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
                                model,
                                X_train,
                                y_train,
                                cv=kfold,
                                scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    #print(cv_results)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    #break
    
#Compare each model
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#Make Predition
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

#Evaluate prediction
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))  