from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB as nb_classifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import pickle
import numpy as np
import pandas as pd

def generateModel(cleanedData,target):

    model = TfidfVectorizer(max_features=1000, min_df=10, max_df=.70, stop_words = stopwords.words('english'))
    print("fittin..")
    model_matrix = model.fit_transform(cleanedData).toarray()
    classifier_matrix = nb_classifier()
    classifier_matrix.fit(model_matrix, target)
    print("fittin done")

    model_file = 'classifier.pickle'
    pickle.dump(classifier_matrix, open(model_file, 'wb'))

    vectorizer = 'vectorizer.pickle'
    pickle.dump(model, open(vectorizer, 'wb'))

    dataframe = pd.DataFrame(model_matrix, columns= model.get_feature_names())
    dataframe['target']= target.tolist()
    print("Loading..")
    dataframe.to_csv(r'E:\myReviewDf.csv')

def test_begins():

    model = pickle.load(open('classifier.pickle','rb'))
    vectorizer= pickle.load(open('vectorizer.pickle', 'rb'))

    text = "p"
    while len(text)> 0:

        text = [text]
        t = vectorizer.transform(text).toarray()
        label = model.predict(t)[0]
        if(label == 0):
            print('Negative')
        else:
            print('Positive')
        text = input("What next?\n")


def gatherData():

    reviewData = load_files('E:/data')
    return reviewData

def cleanData():

    reviewData = gatherData()
    feature, target= reviewData.data, reviewData.target
    #target = target[0:501]
    print(len(target))
    cleanedData=[]
    stemwords= WordNetLemmatizer()

    print('training.. ')
    for i in range(0,len(feature)):
        print("Round ", i, "..")
        cleanedReview = str(feature[i])
        cleanedReview = cleanedReview.replace(r"\n", "")
        cleanedReview = cleanedReview.replace("b\"", "")
        cleanedReview= cleanedReview.lower()
        words = cleanedReview.split(' ')
        for j in range(0,len(words)-1):
            if "n't" in words[j] or "not" in words[j]:
                words[j]= "not_"+words[j+1]
                j+=1
        lemmatized = []

        for word in words:
            lemmatized.append(stemwords.lemmatize(word))

        lemmatized = list(set(lemmatized))
        lemmatized.sort()
        lemmatized = [word for word in lemmatized if not word in stopwords.words()]
        lemmaString = ' '.join(lemmatized)

        cleanedData.append(lemmaString)

    generateModel(cleanedData,target)

if __name__ == '__main__':

    optln = int(input("1.Test or 2.Train\n"))

    if(optln == 1):
        test_begins()
    elif optln == 2:
        cleanData()
