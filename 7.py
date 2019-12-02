#Importing libraries
import pandas as pd

#Importing the dataset
df = pd.read_csv('HeartDisease_Data.csv')
df['Heart Disease'] = df['Heart Disease'].map(dict(Yes=1, No=0))
y = df.iloc[:, -1].values
y=y.reshape(-1,1)
df2 = df.iloc[:,1:-1]
df3 = pd.get_dummies(df2,columns=['Diet','Cholestrol'])
X = df3.values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

#Training the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Import naive bayesian library
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
print(y_predict)

#Import metrics to compare
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)