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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""---ANN!---"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred

for j in range(3):
    if y_pred[j][0]>.5:
        print('True')
    else:
        print('False')