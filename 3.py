# Importing the library
import pandas as pd

# Importing the dataset
df = pd.read_csv('Weather.csv')
df['Play Tennis'] = df['Play Tennis'].map(dict(Yes=1, No=0))
df2 = pd.get_dummies(df,columns=['Temperature']) #drop_first=True
X = df2.iloc[:, 1:4].values
y = df.iloc[:, 5].values
y=y.reshape(-1,1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])
labelencoder_y = LabelEncoder()
# Creating relational order in the variables    
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Training the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Importing the decision tree library and training the agent
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_predict

# Printing the accuracy in the prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)