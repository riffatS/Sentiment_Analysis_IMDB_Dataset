#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# #Data Reading 

# In[2]:


imdbDataset = pd.read_csv(r'IMDB_Dataset.csv')
imdbDataset


# In[3]:


imdbDataset.describe()


# In[4]:


imdbDataset['sentiment'].value_counts()


# #Preprocessing 
# 

# In[7]:


imdbDataset['sentiment']=imdbDataset['sentiment'].map({'positive': 1, 'negative': 0})
stopWords = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()
def preprocessing(content):
    #removing html tags
    content=re.sub('<[^>]*>','',content)
    #remove brackets, integers and alphabets starting of the string 
    content = re.sub(r'[^A-Za-z0-9]+',' ',content)
    #lower the words
    content = content.lower()
    #word stemming using porterStemmer
    ps=nltk.porter.PorterStemmer()
    content= ' '.join([ps.stem(word) for word in content.split(" ")])
    #lemmatize group different words in same form 
    content = [lemmatizer.lemmatize(i) for i in content.split(" ")]
    #pos tagging for verb 
    content = [lemmatizer.lemmatize(i, "v") for i in content]
    content =  " ".join([word for word in content if not word in stopWords])
    return content

imdbDataset['ProcessedReviews'] = imdbDataset.review.apply(lambda x: preprocessing(x))


# In[8]:


#preprocessed data
imdbDataset


# #Splitting dataset

# In[9]:


x = imdbDataset['ProcessedReviews']
y = imdbDataset['sentiment']
x.shape


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# X_train.shape
print("Training example size",X_train.shape)
print("Testing example size",X_test.shape)


# Feature Extraction 

# In[11]:


freqWords = CountVectorizer().fit(imdbDataset['ProcessedReviews'].values.astype('U'))
featureTrain = freqWords.transform(X_train.values.astype('U'))
featureTest = freqWords.transform(X_test.values.astype('U'))


# In[12]:


features = pd.DataFrame(featureTrain.toarray(), columns=freqWords.get_feature_names())
features.sum().sort_values(ascending=True)[:10].plot(kind='barh', title='Top 10 Words',xlabel='freq', ylabel='Words' )


# Algorithm Training

# In[19]:


SVM = SVC()
# fit the model with pre-processed data
SVM.fit(featureTrain, y_train)
#perform classification and prediction on samples in tf_test
predicted = SVM.predict(featureTest)


# In[20]:


#plot confusion matrix for SVM
confusionMatrixSVM = metrics.confusion_matrix(y_test, predicted)
cm = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrixSVM, display_labels = [False, True])
cm.plot(cmap=plt.cm.Blues)
plt.show()


# In[21]:


#print classification report for SVM
print(classification_report(y_test, predicted))


# In[13]:


#logistic regression model
lr=LogisticRegression(max_iter=700,C=1,random_state=42)
lrFit=lr.fit(featureTrain, y_train)
lrPredict=lr.predict(featureTest)


# In[16]:


lrAccuracyScore=accuracy_score(y_test,lrPredict)
print("Logistic Regression Accuracy  :",lrAccuracyScore)


# In[18]:


# Logistic Regression COnfusion Matrix report
confusionMatrixLR = metrics.confusion_matrix(y_test, lrPredict)
cmLR = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrixLR, display_labels = [False, True])
cmLR.plot(cmap=plt.cm.Greens)
plt.show()


# In[23]:


#random forest classifier 
randomForest = RandomForestClassifier(criterion = 'entropy', n_estimators=100)
randomForest.fit(featureTrain, y_train)


# In[24]:


randomForestPred = randomForest.predict(featureTest)


# In[15]:


rfAccuracyScore=accuracy_score(y_test,randomForestPred)
print("Random Forest Accuracy  :",rfAccuracyScore)


# In[22]:


#confusion matrix random forest
confusionMatrixRF = metrics.confusion_matrix(y_test, randomForestPred)
cmRF = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrixRF, display_labels = [False, True])
cmRF.plot(cmap=plt.cm.plasma)
plt.show()

