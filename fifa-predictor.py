#FIFA Predictor

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset= pd.read_csv('WorldCupMatches.csv')
X= dataset.iloc[:, [0,2,5,8,11,12]].values
#y= dataset.iloc[:, 20].values
y= dataset.iloc[:,-1].values

year_list= np.unique(X[:,0].astype(str))
stage_list= [x.lower() for x in np.unique(X[:,1].astype(str))]
hometeam_list= [x.lower() for x in np.unique(X[:,2].astype(str))]
awayteam_list= [x.lower() for x in np.unique(X[:,3].astype(str))]

# Encoding categorical features
# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_year= LabelEncoder()
labelencoder_stage= LabelEncoder()
labelencoder_hometeam= LabelEncoder()
labelencoder_awayteam= LabelEncoder()

X[:,0] = labelencoder_year.fit_transform(X[:,0])
X[:,1] = labelencoder_stage.fit_transform([x.lower() for x in X[:,1].astype(str)])
X[:,2] = labelencoder_hometeam.fit_transform([x.lower() for x in X[:,2].astype(str)])
X[:,3] = labelencoder_awayteam.fit_transform([x.lower() for x in X[:,3].astype(str)])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X= imputer.fit_transform(X)        
            
# One Hot Encoding
onehotencoder= OneHotEncoder(categorical_features=[0,1,2,3])
X= onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X = np.delete(X, [0,20,37,115], axis=1)
    
# Preprocessing the output data
# Encoding Categorical features
from sklearn.preprocessing import LabelEncoder
labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)


# - Function to preprocess the input data
def input_preprocessing(X):
    
    # Encoding categorical features
    # Label Encoding
    X[0] = labelencoder_year.transform([X[0]])
    X[1] = labelencoder_stage.transform([X[1]])
    X[2] = labelencoder_hometeam.transform([X[2]])
    X[3] = labelencoder_awayteam.transform([X[3]])
    
    #One Hot Encoding
    X= onehotencoder.transform([X]).toarray()
    
    #Avoiding dummy variable trap
    X= np.delete(X, [0,20,37,115], axis=1)
    
    # Feature Scaling
    X= sc.transform(X)
    
    return X    

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Activation

#Initializing the ANN
classifier= Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=64, activation='relu', kernel_initializer='uniform', input_dim=196))

#Adding the second hidden layer
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))

#Adding the third hidden layer
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))

#Adding the activation layer
classifier.add(Activation('softmax'))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=20, epochs=250)

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred= (y_pred> 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

# Printing the accuracy of the model
accuracy_test= (cm[0,0]+cm[1,1])/np.sum(cm)*100
print('\nThe accuracy of the predictions by the ANN model on the test set = {}'.format(accuracy_test))


#Implementing the model on user input
print('------------------PREDICT MATCH RESULTS------------------')
print('This predictor predicts FIFA match results for popular teams between 1930-2014.')
print('Some of the stages are Group A-H, Quarter finals, Semi finals, Finals. For others, refer to the attached dataset.' )

while True:
    test_year = input('\nEnter the year of match: ')
    if test_year in year_list:
        break
    print('FIFA did not take place in year {}. Please try again.'.format(test_year))

while True:
    test_stage = input('\nEnter the stage of match: ').lower()
    if test_stage in stage_list:
        break
    print('Stage \'{}\' not listed. Please try again.'.format(test_stage))

while True:
    test_hometeam = input('\nEnter name of home team: ').lower()
    if test_hometeam in hometeam_list:
        break
    print('Home team not listed. Please try again.')
    
while True:
    test_awayteam = input('\nEnter name of away team: ').lower()
    if test_awayteam in hometeam_list:
        break
    print('Away team not listed. Please try again.')

test_hometeamgoals = input('\nEnter the approximate number of goals {} scores before half-time: '
                           .format(test_hometeam.title()))
test_awayteamgoals = input('\nEnter the approximate number of goals {} scores before half-time: '
                           .format(test_awayteam.title()) )

#Passing input variables for processing in the form required by the model
test_X = input_preprocessing(np.array([int(test_year), test_stage, test_hometeam, test_awayteam, 
                          int(test_hometeamgoals), int(test_awayteamgoals)], dtype= object))
test_y = classifier.predict(test_X)


print('\nThe probability that the home team {} wins over the away team {} ={}%'
      .format(test_hometeam.title(), test_awayteam.title(), (round(float(test_y), 4))*100))




