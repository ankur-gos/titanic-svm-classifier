'''
    svm.py
    Use svm to classify the titanic dataset
    Ankur Goswami, agoswam3@ucsc.edu
'''

import pandas as pd
import matplotlib as plt
from sklearn.svm import SVC

def read_data(file):
    f = pd.read_csv(file)
    return f

def extract_labels(df):
    return df['Survived']

def gender_transform(gender):
    if gender == 'male':
        return 0
    return 1


def prepare_df(df):
    labels = None
    if 'Survived' in df:
        labels = extract_labels(df)
        del df['Survived']
    del df['Embarked']
    del df['Ticket']
    del df['Cabin']
    del df['Name']
    del df['PassengerId']
    del df['Fare']
    df['Sex'] = df['Sex'].apply(gender_transform)
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].apply(lambda val: mean_age if pd.isnull(val) else val)
    return df, labels

def run_svm(train_file):
    clf = SVC(C=1.5, gamma=0.16)
    df = read_data(train_file)
    example_count = df.shape[0]
    df, labels = prepare_df(df)
    sep = int(0.6*example_count)
    train_df = df.iloc[:sep]
    train_labels = labels.iloc[:sep]
    validate_df = df.iloc[sep:]
    validate_labels = labels.iloc[sep:]
    clf.fit(train_df, train_labels)
    bias = clf.score(train_df, train_labels)
    variance = clf.score(validate_df, validate_labels)
    print bias, variance
    return clf
    # print bias, variance

def predict(test_file, clf):
    init_df = read_data(test_file)
    df, labels = prepare_df(init_df)
    predictions = clf.predict(df)
    pdf = pd.Series(predictions)
    pdf.index += 892
    return pdf

clf = run_svm('train.csv')
pdf = predict('test.csv', clf)
pdf.to_csv('results.csv')
