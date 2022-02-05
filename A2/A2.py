# import libraries
import sagemaker
import boto3
from sagemaker import get_execution_role

import numpy as np
import pandas as pd
from time import gmtime, strftime
import os, time
from matplotlib import pyplot 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from keras.utils import np_utils

# set up AWS environments
region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')

role = get_execution_role()
print(role)

bucket = "sagemaker-michaelwu-ma5852"
subfolder = 'A2'
input_file_name = 'diabetic_data.csv'
input_file_path = f's3://{bucket}/{subfolder}/{input_file_name}'


# load data
df_raw = pd.read_csv(input_file_path)

# proposed predictors (features)
features = ['max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']

X = df_raw[features]
y = df_raw[['readmitted']]
num_class = len(df_raw['readmitted'].unique())

'''
    one hot encode categorical data
    this is because all the predictors and target variable are categorical data type
'''
# prepare input data
def prepare_inputs(X):
    ohe = OneHotEncoder()
    ohe.fit(X)
    X_enc = ohe.transform(X)
    return X_enc

# prepare target
def prepare_target(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    y_enc = np_utils.to_categorical(y_enc, num_class)
    return y_enc

X_enc = prepare_inputs(X)
y_enc = prepare_target(y)


'''
    train test split
'''
X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=1234) # 80/20 split


'''
    build baseline MLP NN model
'''
# define model structure
model_baseline = tf.keras.models.Sequential()
model_baseline.add(tf.keras.layers.Dense(58, activation='relu')) # 2/3 of the number of inputs (83) + number of outputs (3, as there are 3 classes) ~ 58
model_baseline.add(tf.keras.layers.Dense(29, activation='relu')) # half number of neurons as the first hidden layer
model_baseline.add(tf.keras.layers.Dense(num_class, activation='softmax')) # number of neurons at the output layer = number of classes of the target

# compile the model
model_baseline.compile(optimizer="adam", loss="categorical_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# train model
start_time = time.time()
history_baseline = model_baseline.fit(x=X_train, y=y_train, epochs=300, verbose=True, validation_data=(X_test, y_test), batch_size=60)
print(f'Baseline NN model took {time.time() - start_time} to complete training...')

#display a summary of the model
model_baseline.summary()

# plot model performance
pd.DataFrame(history_baseline.history).plot(figsize=(8,5))
pyplot.grid(True)
pyplot.gca().set_ylim(0,1) #set the limits of the y axis


'''
    evaluate model on test set
'''
score = model_baseline.evaluate(X_test, y_test, verbose=0)
print('Test loss    :', score[0])
print('Test accuracy:', score[1])
print('Test precision:', score[2])
print('Test recall:', score[3])