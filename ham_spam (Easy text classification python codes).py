# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:55:33 2021

@author: Soumya PC
"""

import pandas as pd
Messages = pd.read_csv("ham_spam.csv", encoding= "ISO-8859-1")

#Data cleaning & preprocessing step
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0, len(Messages)):
    review = re.sub('[^a-zA-Z]',' ',Messages['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#creating a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y= pd.get_dummies(Messages['type'])
y=y.iloc[:,1].values

#Train- Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

#training the model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)


#Cross table
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test, y_pred)
 
#Accuracy test
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)

 #Here we are getting the accuracy score of 98%