import numpy as np
import pandas as pd
import pickle
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt  # plotting
from mpl_toolkits.mplot3d import Axes3D
from google.cloud import storage
import slack

from services.plots.plots import confusion_matrix_plot

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, roc_auc_score
from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATASET_SIZE = 100
SAFE_RATIO = 0.5
DEBUG = False
LOCAL = False
SLACK_NOTIFY = False
SLACK_CHANNEL = ''

if SLACK_NOTIFY:
    BOTKEY = os.environ['SLACK_BOT_TOKEN']

VERSION = '1_1'

if not DEBUG:
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn


def send_slack_message(message):
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    client = slack.WebClient(token=slack_token)

    client.chat_postMessage(
        channel=SLACK_CHANNEL,
        text=message
    )

def print_report(name, train_score, test_score, m_mean, best_para):
    print('Training: ', name)
    print('-----------------------------')
    print('Accuracy training score: ', train_score)
    print('Accuracy test score: ', test_score)
    print('Mean squared error: ', m_mean)
    print('Best parameters: ', best_para)
    print()

def save_results(names, mean_errors, train_scoring, test_scoring, aucrocs, best_params, train_times):
    result_obj = {
        'names': names,
        'model_mean_errors': mean_errors,
        'model_train_scoring': train_scoring,
        'model_test_scoring': test_scoring,
        'best_parameters': best_params,
        'aucrocs': aucrocs,
        'times': training_times
    }

    with open('data/results/results.pkl', 'wb') as result_out:
        pickle.dump(result_obj, result_out)

    fig = plt.figure(figsize=(10, 10))
    s = fig.add_subplot(111)
    s.bar(result_obj['names'], result_obj['model_train_scoring'], label='Train score')
    s.bar(result_obj['names'], result_obj['model_test_scoring'], label='Test score')
    s.set_ylim(0.5, 1.1)
    s.set_xlabel('Models', fontsize=20)
    s.set_ylabel('Accuracy %', fontsize=20)
    s.set_title('Test/Train accuracy', fontsize=30)
    s.legend()
    fig.savefig('./data/results/test_train')

    fig = plt.figure(figsize=(10, 10))
    s = fig.add_subplot(111)
    s.bar(result_obj['names'], result_obj['times'], label='Training time')
    s.set_ylim(0.5, 1.1)
    s.set_xlabel('Models', fontsize=20)
    s.set_ylabel('min', fontsize=20)
    s.set_title('Training Time (min)', fontsize=30)
    s.legend()
    fig.savefig('./data/results/training_time')

'''
    fig = plt.figure(figsize=(10, 10))
    s = fig.add_subplot(111)
    s.bar(result_obj['names'], result_obj['aucrocs'], label='AUCROC score')
    s.set_ylim(0.5, 1.1)
    s.set_xlabel('Models', fontsize=20)
    s.set_ylabel('Aucroc', fontsize=20)
    s.set_title('AUCROC score', fontsize=30)
    s.legend()
    fig.savefig('./data/results/aucrocs')
'''


'''-----------------------------'''
''' Import Dataset              '''
'''-----------------------------'''
dataset = pd.read_csv('./Data/findata.csv')

'''-----------------------------'''
''' Data exploration            '''
'''-----------------------------'''

# dataframe head

# number of instances and attributes

# feature data info

# target data info

# dataframe describe

# dataframe parameter overview

# correlation heatmap

# correlation scatter plots (selected)

'''-----------------------------'''
''' Feature Engineering         '''
'''-----------------------------'''

# Drop non unique attributes and ones who could lead to overfitting
dataset = dataset.drop(['zipcodeOri', 'zipMerchant', 'customer'], axis=1)

# get a higher ratio of frauds
SAFE_RATIO = 0.5
fraud_df = dataset.loc[dataset['fraud'] == 1]
safe_amount = int((len(fraud_df)*2)*SAFE_RATIO)
safe_df = dataset.loc[dataset['fraud'] == 0][:safe_amount]
norm_distri_df = pd.concat([fraud_df, safe_df])
dataset = norm_distri_df.sample(frac=1, random_state=42)

#dataset['customer'] = dataset['customer'].replace('\'','', regex=True)
dataset['age'] = dataset['age'].replace('\'','', regex=True)
dataset['gender'] = dataset['gender'].replace('\'','', regex=True)
dataset['merchant'] = dataset['merchant'].replace('\'','', regex=True)
dataset['category'] = dataset['category'].replace('\'','', regex=True)


# Label encoder for categorical values
label_encoder = LabelEncoder()
dataset['gender'] = label_encoder.fit_transform(dataset['gender'])
dataset['merchant'] = label_encoder.fit_transform(dataset['merchant'])
dataset['category'] = label_encoder.fit_transform(dataset['category'])
dataset['age'] = label_encoder.fit_transform(dataset['age'])

# Value still string in age, replace by highest integer
dataset['age'] = dataset['age'].replace('U',7, regex=True)


# setting all int values to float
for column in range(dataset.shape[1]):
    dataset[:column] = dataset[:column] * 1.


# split into features and target
if DEBUG:
    X = dataset.iloc[:DATASET_SIZE,:-1].values
    Y = dataset.iloc[:DATASET_SIZE,6]
else:
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 6]


# Dummy Encoding
# Because one category should not be greater then any other
sc_X = StandardScaler()
hot_encoder = OneHotEncoder(categorical_features=[1,2,4])
X = hot_encoder.fit_transform(X).toarray()

# Label encoder for Y
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)


'''-----------------------------'''
''' Data Seqrigation (splits)   '''
'''-----------------------------'''

# x_test, x_train, y_test, y_train
# splitting dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


'''-----------------------------'''
''' Define hyperparamets        '''
'''-----------------------------'''

# store all hyperparams in object
KNN_hyperparams = {
    'kneighborsclassifier__n_neighbors': [3, 5, 8, 10, 15, 20],
    'kneighborsclassifier__p': [1, 2, 3, 4, 5]
}

NB_hyperparams = {}

tree_hyperparams = {
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__max_depth': [1, 2, 4, 8, 16, 32],
    'decisiontreeclassifier__max_features': [1, 2, 4, 6, 8, 12]
}

SVC_hyperparams = {
    'svc__kernel': ['rbf', 'sigmoid', 'linear'],
    'svc__gamma': [1e-2, 1e-3, 1e-4, 1e-5],
    'svc__C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
}

neural_hyperparams = {
    'mlpclassifier__solver': ['adam', 'lbfgs'],
    'mlpclassifier__max_iter': [50, 100, 150, 200, 300, 400, 600],
    'mlpclassifier__alpha': 10.0 ** -np.arange(1, 5),
    'mlpclassifier__hidden_layer_sizes': np.arange(10, 15)
}

ada_hyperparams = {
    'adaboostclassifier__n_estimators': [50, 100, 250, 500, 750, 1000],
    'adaboostclassifier__learning_rate': 10.0 ** -np.arange(0, 4),

}

LogiRegressor_hyperparams = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

RandForest_hyperparams = {
    'randomforestclassifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'randomforestclassifier__max_features': ['auto', 'sqrt'],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__n_estimators': [100, 200, 400, 800, 1200, 1600, 2000]
}


# define lists for train_accuracies, test_accuracies, mean_errors, names and best_params
model_train_scoring = []
model_test_scoring = []
model_mean_error = []
names = []
best_param = []
aucrocs = []
training_times = []

'''-----------------------------'''
''' Define all classifiers      '''
'''-----------------------------'''

# Init model array
models = []
models.append(('KNN', KNeighborsClassifier(), KNN_hyperparams))
models.append(('NaiveBayes', GaussianNB(), NB_hyperparams))
models.append(('DecisionTree', DecisionTreeClassifier(random_state=0), tree_hyperparams))
models.append(('SupportVectorMachine', SVC(random_state=0), SVC_hyperparams))
models.append(('NeuralNetwork', MLPClassifier(random_state=0, verbose=False), neural_hyperparams))
#models.append(('AdaBoost', AdaBoostClassifier(random_state=0), ada_hyperparams))
#models.append(('LogisticRegression', LogisticRegression(random_state=0), LogiRegressor_hyperparams))
#models.append(('RandomForest', RandomForestClassifier(random_state=0), RandForest_hyperparams))

'''-----------------------------'''
''' Training All                '''
'''-----------------------------'''

def save_model(model, model_name):
    if not os.path.isdir('./data/models'):
        os.mkdir('./data/models')

    with open('./data/models/{}.pkl'.format(model_name), 'wb') as m_file:
        pickle.dump(model, m_file)


for name, model, params in models:
    start_time = time.time()
    pipeline = make_pipeline(preprocessing.StandardScaler(), model)
    grid_search = GridSearchCV(pipeline, params, cv=10, verbose=0, error_score=np.nan)
    grid_search.fit(X_train, Y_train)
    end_time = time.time()

    save_model(grid_search, name)

    train_pred = grid_search.predict(X_train)
    test_pred = grid_search.predict(X_test)
    #rocauc_score = roc_auc_score(Y_test, test_pred)

    names.append(name)
    model_train_scoring.append(r2_score(Y_train, train_pred))
    model_test_scoring.append(r2_score(Y_test, test_pred))
    model_mean_error.append(mean_squared_error(Y_test, test_pred))
    best_param.append(grid_search.best_params_)
    #aucrocs.append(rocauc_score)
    training_times.append(end_time - start_time)

    print_report(name, r2_score(Y_train, train_pred), r2_score(Y_test, test_pred), mean_squared_error(Y_test, test_pred), grid_search.best_params_)

    confusion_matrix_plot(name, Y_test, test_pred)


'''-----------------------------'''
''' Save all results and models '''
'''-----------------------------'''

def save_all_GCP():
    BUCKET = storage.Client().bucket('birtai_storage')

    model_dir = './data/results/'
    result_dir = './data/models/'
    version = 'v' + str(VERSION)
    bucket_head_dir = 'CreditCardFraud_{}'.format(version) + '/'

    for file in os.listdir(model_dir):
        local_file = model_dir + file
        blob = BUCKET.blob(bucket_head_dir + 'models/' + file)
        blob.upload_from_filename(local_file)

    for file in os.listdir(result_dir):
        local_file = result_dir + file
        blob = BUCKET.blob(bucket_head_dir + 'models/' + file)
        blob.upload_from_filename(local_file)


save_results(names, model_mean_error, model_train_scoring, model_test_scoring, None, best_param, training_times)

if not LOCAL:
    save_all_GCP()



