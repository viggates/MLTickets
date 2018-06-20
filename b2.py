import sys
import nltk
import numpy as np
import pandas as pd
import pickle
# from helpers import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
sys.path.append(".")
sys.path.append("..")
# Use the Azure Machine Learning data preparation package
# from azureml.dataprep import package


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


import pudb;pu.db
column_to_predict = "HelpTopic"
# Supported datasets:
# ticket_type
# business_service
# category
# impact
# urgency
# sub_category1
# sub_category2

classifier = "NB"  # Supported algorithms # "SVM" # "NB"
use_grid_search = False  # grid search is used to find hyperparameters. Searching for hyperparameters is time consuming
remove_stop_words = True  # removes stop words from processed text
stop_words_lang = 'english'  # used with 'remove_stop_words' and defines language of stop words collection
use_stemming = True  # word stemming using nltk
fit_prior = False  # if use_stemming == True then it should be set to False ?? double check
min_data_per_class = 0  # used to determine number of samples required for each class.Classes with less than that will be excluded from the dataset. default value is 1

if __name__ == '__main__':

    # TODO Add download dataset
     
    # loading dataset from dprep in Workbench    
    # dfTickets = package.run('AllTickets.dprep', dataflow_idx=0) 

    # loading dataset from csv
    #dfTickets = pd.read_csv(
    #    '../input.csv',
    #    dtype=str
    #)  

    dfTickets = pd.read_excel(
        '../SampleInput.xlsx',
        sheetname='TicketInputData'
    )  
    dfTickets = dfTickets.filter(['Title','HelpTopic'])
    text_columns = "Title"  # "title" - text columns used for TF-IDF
    
    # Removing rows related to classes represented by low amount of data
    print("Shape of dataset before removing classes with less then " + str(min_data_per_class) + " rows: "+str(dfTickets.shape))
    print("Number of classes before removing classes with less then " + str(min_data_per_class) + " rows: "+str(len(np.unique(dfTickets[column_to_predict]))))
    bytag = dfTickets.groupby(column_to_predict).aggregate(np.count_nonzero)
    tags = bytag[bytag.Title > min_data_per_class].index
    dfTickets = dfTickets[dfTickets[column_to_predict].isin(tags)]

    #x = dfTickets.groupby(['TicketNumber', 'Title']).aggregate(np.count_nonzero)
    #xt = x[x.Resolution > min_data_per_class].index
    #dfTickets = dfTickets[dfTickets[column_to_predict].isin(xt)]
    # one more test
    dfTickets = dfTickets[dfTickets.Title.str.contains("Ticket") == False]
    dfTickets = dfTickets[dfTickets.Title.str.contains("Status") == False]
    dfTickets = dfTickets[dfTickets.Title.str.contains("Collaborators") == False]
    #Collaborators
    print(dfTickets.values)
    dfTickets[column_to_predict].dropna(how='any')
    dfTickets = dfTickets.reset_index(drop=True)
    #dfTickets = dfTickets[dfTickets.Title.str.contains("nan") == False]
    
    print(
        "Shape of dataset after removing classes with less then "
        + str(min_data_per_class) + " rows: "
        + str(dfTickets.shape)
    )
    print(
        "Number of classes after removing classes with less then "
        + str(min_data_per_class) + " rows: "
        + str(len(np.unique(dfTickets[column_to_predict])))
    )

    labelData = dfTickets[column_to_predict]
    data = dfTickets[text_columns]

    # added NaN
    #labelData = labelData.fillna(' ')
    #data = data.fillna(' ')

    # Split dataset into training and testing data
    #train_data, test_data, train_labels, test_labels = train_test_split(
    #    data, labelData, test_size=0.2
    #)  # split data to train/test sets with 80:20 ratio


    # Extracting features from text
    # Count vectorizer
    if remove_stop_words:
        count_vect = CountVectorizer(stop_words=stop_words_lang)
    elif use_stemming:
        count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)
    else:
        count_vect = CountVectorizer()


    text_clf = Pipeline([
            ('vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB(fit_prior=fit_prior))
    ])


    kf = KFold(n_splits=10)
    kf.get_n_splits(data)
    for train_index, test_index in kf.split(data):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labelData[train_index], labelData[test_index]

#        x = text_clf.fit_transform(df['Review'].values.astype('U'))
        text_clf = text_clf.fit(train_data.values.astype('U'), train_labels.values.astype('U'))

        print("Evaluating model")
        # Score and evaluate model on test data using model without hyperparameter tuning
        predicted = text_clf.predict(test_data.values.astype('U'))
        prediction_acc = np.mean(predicted == test_labels)
        print("Confusion matrix without GridSearch:")
        print(metrics.confusion_matrix(test_labels, predicted))
        print("Mean without GridSearch: " + str(prediction_acc))

        print(classification_report(test_labels, predicted,
                                target_names=np.unique(test_labels)))


    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import matplotlib
    mat = confusion_matrix(test_labels, predicted)
    plt.figure(figsize=(4, 4))
    sns.set()
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    # Save confusion matrix to outputs in Workbench
    # plt.savefig(os.path.join('.', 'outputs', 'confusion_matrix.png'))
    plt.show()
    """
    # Printing classification report
    # Use below line only with Jupyter Notebook
    # Use with Workbench
    if use_grid_search:
        pickle.dump(
            gs_clf,
            open(os.path.join(
                '.', 'outputs', column_to_predict+".model"),
                'wb'
            )
        )
    else:
        pickle.dump(
            text_clf,
            open(os.path.join(
                '.', 'outputs', column_to_predict+".model"),
                'wb'
            )
        )
