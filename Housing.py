from pandas import read_csv
from pandas.plotting import scatter_matrix
#from matplotlib import pyplot
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression

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


dataset = read_csv('house_data.csv')

#Summarize the dataset
    #shape
print(dataset.shape)
    #head
print(dataset.head(10))
    #description
print(dataset.describe()) 

'''
plt.scatter(dataset.price, dataset.date)
plt.title("Price Vs Square Feet")
plt.show()
'''

y = dataset['price']
conv_date = [1 if values == 2014 else 0 for values in dataset.date]
dataset['date'] = conv_date
x = dataset.drop(['id','price'], axis=1)
X_train, X_validation, y_train, y_validation = train_test_split(
                                                                x,
                                                                y,
                                                                test_size = 0.10,
                                                                random_state = 2)

reg = LinearRegression()
reg.fit(X_train, y_train)
score = reg.score(X_validation, y_validation)

print(score)