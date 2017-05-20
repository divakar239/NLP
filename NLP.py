#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:11:51 2017

@author: DK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

#importing dataset
dataset=pd.read_csv('/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/NLP/Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Cleaning text:
#1. Getting rid of everything that's not characters
#2. Convert all letters to lowercase
#3. Converting review to a list from a string
#4. Stemming
#5. Populating review again with only those terms that are not part of the stopwords package
#6. Adding the string to a list called corpus
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)               #review turned back to a string from a list keeping all the spaces intact
    corpus.append(review)                   #corpus is a list which is populated with the string review as its first element
             

#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,1].values          



#NOTE: Since the machine is to decide whether a review is positive or negative it is a clasification problem, hence we can use any classification model

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and plotting it
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()


sn.set(font_scale=1.4)#for label size
sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size
accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
print("the accuracy of the model is %f",accuracy*100)




