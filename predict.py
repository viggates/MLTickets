import sys
import os
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time
import pickle
import re
# Use with Azure Web Apps
sys.path.append(".")
sys.path.append("..")



def predictall(model_ticket_type, description):
    ts = time.gmtime()
    description = preprocess_data(description)

    predicted_ticket_type = model_ticket_type.predict([description])[0]
    print("predicted ticket_type: "+str(predicted_ticket_type))


def category1():
    ts = time.gmtime()
    print(request)
    print(request.json)
    if not request.json or 'description' not in request.json:
        abort(400)
    description = request.json['description']
    print(description)

    predicted = model_category.predict([description])
    print("Predicted: "+str(predicted))

    ts = time.gmtime()
    return jsonify({"category": predicted[0]})


def tickettype():
    ts = time.gmtime()
    print(request)
    print(request.json)
    if not request.json or 'description' not in request.json:
        abort(400)
    description = request.json['description']
    print(description)

    predicted = model_ticket_type.predict([description])
    print("Predicted: " + str(predicted))

    ts = time.gmtime()
    return jsonify({"ticket_type": predicted[0]})


# Data prep - much to improve :)
regexArr1 = []
regexArr2 = []


def getRegexList1():
    regexList = []
    regexList += ['From:(.*)']  # from line
    regexList += ['Sent:(.*)']  # sent to line
    regexList += ['Received:(.*)']  # received data line
    regexList += ['To:(.*)']  # to line
    regexList += ['CC:(.*)']  # cc line
    regexList += ['https?:[^\]\n\r]+']  # https & http
    regexList += ['Subject:']
    regexList += ['[\w\d\-\_\.]+@[\w\d\-\_\.]+']  # emails
    return regexList


def getRegexList2():
    regexList = []
    regexList += ['From:']  # from line
    regexList += ['Sent:']  # sent to line
    regexList += ['Received:']  # received data line
    regexList += ['To:']  # to line
    regexList += ['CC:']  # cc line
    regexList += ['The information(.*)infection']  # footer
    regexList += ['Endava Limited is a company(.*)or omissions']  # footer
    regexList += ['The information in this email is confidential and may be legally(.*)interference if you are not the intended recipient']  # footer
    regexList += ['\[cid:(.*)]']  # images cid
    regexList += ['https?:[^\]\n\r]+']  # https & http
    regexList += ['Subject:']
    regexList += ['[\w\d\-\_\.]+@[\w\d\-\_\.]+']  # emails
    regexList += ['[\\r]']  # \r\n
    regexList += ['[\\n]']  # \r\n

    regexList += ['^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$']
    regexList += ['[^a-zA-Z]']

    return regexList


def preprocess_data(data):
    print(data)
    content = data.lower()
    content = content.split('\\n')

    for word in content:
        for regex in regexArr1:
            word = re.sub(regex.lower(), ' ', word)

    print(content)
    content = "".join(content)
    print(content)

    for regex in regexArr2:
        content = re.sub(regex.lower(), ' ', content)
    print(content)

    return content


if __name__ == '__main__':
    print("started")
    # Loading models
    model_ticket_type = pickle.load(open("outputs/Resolution.model", "rb"))
    description = input()
    import pudb;pu.db
    predictall(model_ticket_type, description)


