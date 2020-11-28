from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
import nltk
import re

def gatherData():

    reviewData = load_files('E:/txt_sentoken_dum')
    return reviewData

def cleanData():

    reviewData = gatherData()
    feature, target= reviewData.data, reviewData.target
    cleanedData=[]
    stemwords= WordNetLemmatizer()

    for i in range(0,len(feature)):

        cleanedReview = re.sub(r'\W+',' ',str(feature[i]))
        cleanedReview = re.sub(r'\s+[a-zA-Z]\s+',' ',cleanedReview)
        cleanedReview = re.sub(r'\^[a-zA-Z]\s+',' ',cleanedReview)
        cleanedReview= re.sub(r'^b\s+','',cleanedReview)
        cleanedReview= cleanedReview.lower()
        words = cleanedReview.split(' ')
        lemmatized = []

        for word in words:
            lemmatized.append(stemwords.lemmatize(word))

        lemmatized = list(set(lemmatized))
    #    lemmatized.sort()
        lemmaString = ' '.join(lemmatized)
        cleanedData.append(lemmaString)
        lemmatized = []

if __name__ == '__main__':

    cleanData()
