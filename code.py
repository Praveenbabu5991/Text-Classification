# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:38:02 2018

@author: Bolt
"""
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')


#importing the dataset
reviews=load_files('tokens/')
X,y=reviews.data,reviews.target

#storing as pickle file
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#unpickling the dataset
with open ('X.pickle','rb') as f:
    X=pickle.load(f)
    
with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
#creating the corpus #corpus list of documents
corpus=[]
for i in range(0,len(X)):
    review=re.sub(r'\W',' ',str(X[i]))#removing non words
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)# removing single words
    review=re.sub(r'^[a-z]\s+',' ',review)#single word at front
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)

#creating   BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=1400,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
#remove (mindf) words appearing  below 3,removing(maxdf) words appearing more than 0.6  ,maximum 1400 character among the rest
X=vectorizer.fit_transform(corpus).toarray()    


#converting bow to tfidf
from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()
****************************or*****************
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer =TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()
************************************************
#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#prediction
y_pred=classifier.predict(X_test)

#evaluation
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#importing and using the model
#pickling the classifier 
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)


#unpickling the classifier
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)
    
sample=["i hate mumbai"]
sample=tfidf.transform(sample).toarray()
print(clf.predict(sample))
    
    
    
    
    
    
    
    