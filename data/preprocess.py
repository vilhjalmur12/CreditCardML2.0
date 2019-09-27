import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sc_X = StandardScaler()
hot_encoder = OneHotEncoder(categorical_features=[2,3,4])

def dataPreprocess():
    ###########################################
    #   Data Preprocessing
    ###########################################

    # Import dataset
    dataset = pd.read_csv('./Data/findata.csv')

    # There is shortening data there
    X = dataset.iloc[:20000, :-1].values
    Y = dataset.iloc[:20000, 9]

    # Dropping attributes which could lead to overfitting
    # In this case some attributes only have 1 unique value
    tmp_X = X[:, [0,2,3,5,7,8]]
    X = tmp_X

    # Categorical data
    # Some attributes are still categorical and we need to fit transform the data to numbers
    # Categorical attributes: gender column 2, merchant column 3 and category column 4
    label_encoder_X = LabelEncoder()
    X[:, 2] = label_encoder_X.fit_transform(X[:, 2])
    X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
    X[:, 4] = label_encoder_X.fit_transform(X[:, 4])

    # cleaning up commas in strings
    tmp_X = []
    for column in X:
        tmp = []
        for item in column:
            if type(item) is str:
                tmp.append(item.replace('\'', ''))
            else:
                tmp.append(item)
        tmp_X.append(tmp)
    X = tmp_X



    # There is a problem with encoding one value in the gender column so we need to do it manually
    for column in range(0, len(X)):
        if X[column][1] == 'U':
            X[column][1] = 7



    # Change all values in X to float
    tmp_X = []
    for column in X:
        tmp_X.append([float(item) for item in column])
    X = np.array(tmp_X)


    # Dummy Encoding
    # Because one category should not be greater then any other
    X = hot_encoder.fit_transform(X).toarray()


    # Label encoder for Y
    label_encoder_Y = LabelEncoder()
    Y = label_encoder_Y.fit_transform(Y)


    # splitting dataset into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


    # Feature scaling
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return { 'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}



# processes the new input data for prediction
def predDataProcess(data):

    X = np.array(data)

    # Dummy encoding with existing encoder
    X = hot_encoder.transform(X).toarray()

    # Feature scaling with existing encoder
    X = sc_X.transform(X)

    return X

