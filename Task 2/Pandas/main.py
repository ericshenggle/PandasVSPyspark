#!/usr/bin/env python
# coding: utf-8

# ---

# ### Description:
# 
# The dataset is comprised of tab-separated files with phrases from the IMDB Movie Ratings. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.
# 
# train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
# test.tsv contains just phrases. You must assign a sentiment label to each phrase.
# The sentiment labels are:
# 
# 0 - negative
# 1 - positive
# 
# ### Objective:
# - Understand the Dataset & cleanup (if required).
# - Build classification models to predict the ratings of the movie.
# - Compare the evaluation metrics of vaious classification algorithms.

#  

# ---

# ## <center>1. Data Exploration

# In[3]:


#Importing the necessary librarires

import os
import math
import nltk
import scipy
import string
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, \
roc_auc_score, roc_curve, precision_score, recall_score

import warnings 
warnings.filterwarnings('ignore')

import datetime
start_time = datetime.datetime.now()

task_names = ['Load', 'Preprocess', 'LR Train', 'LR Evaluate', 'NB Train', 'NB Evaluate', 'DT Train', 'DT Evaluate', 'RF Train', 'RF Evaluate']
task_times = []


# In[4]:


#Importing the dataset

df = pd.read_csv('../Datasets/movie.csv', header=0)
target = 'label'
df.reset_index(drop=True, inplace=True)
original_df = df.copy(deep=True)

print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()


# **Inference:** The stats seem to be fine, let us gain more undestanding by visualising the dataset.

# ---

# ## <center> 2. Data Preprocessing

# In[5]:


#Check for empty elements

print(df.isnull().sum())
print('\n\033[1mInference:\033[0m The dataset doesn\'t have any null elements')


# In[6]:


#Removal of any Duplicate rows (if any)

counter = 0
r,c = original_df.shape

df1 = df.drop_duplicates()
df1.reset_index(drop=True, inplace=True)

# if df1.shape==(r,c):
#     print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
# else:
#     print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {r-df1.shape[0]}')


# In[7]:


#Filtering the text

import nltk
import string
from tqdm import tqdm
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df = df1.copy()

def preprocessor(text):
    text = text.lower()
    text = ''.join([i for i in text if i in string.ascii_lowercase+' '])
    text = ' '.join([PorterStemmer().stem(word) for word in text.split()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

#with Pool(4) as p:
#    df['text'] = list(tqdm(p.imap(preprocessor, range(df.shape[0]))))
for i in tqdm(range(df.shape[0])):
    df.loc[i,'text'] = preprocessor(df['text'][i])

#from tqdm.contrib.concurrent import process_map 

#df['text'] = process_map(tqdm(preprocessor, df['text'], max_workers=8))

#for i in tqdm()

# df.head()


# **Inference:** The text is now clean up with the removal of all punctuations, stopwords & stemming. 

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
def tokenizer(text):
        return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True)
y=df.label.values
x=tfidf.fit_transform(df.text)


# ---

# ## <center> 3. Predictive Modeling

# In[9]:


#Splitting the data intro training & testing sets

X = df.drop([target],axis=1)
Y = df[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

print('Original set  ---> ',X.shape,Y.shape,'\nTraining set  ---> ',Train_X.shape,Train_Y.shape,'\nTesting set   ---> ', Test_X.shape,'', Test_Y.shape)


# In[10]:


#Let us create first create a table to store the results of various models 

Evaluation_Results = pd.DataFrame(np.zeros((4,5)), columns=['Accuracy', 'Precision','Recall','F1-score','AUC-ROC score'])
Evaluation_Results.index=['Logistic Regression (LR)','Decision Tree Classifier (DT)','Random Forest Classifier (RF)','Naïve Bayes Classifier (NB)']
Evaluation_Results


# In[11]:


#Let us define functions to summarise the Prediction's scores .

#Classification Summary Function
def Classification_Summary(pred,pred_prob,i):
    Evaluation_Results.iloc[i]['Accuracy']=round(accuracy_score(Test_Y, pred),3)*100   
    Evaluation_Results.iloc[i]['Precision']=round(precision_score(Test_Y, pred),3)*100 #, average='weighted'
    Evaluation_Results.iloc[i]['Recall']=round(recall_score(Test_Y, pred),3)*100 #, average='weighted'
    Evaluation_Results.iloc[i]['F1-score']=round(f1_score(Test_Y, pred),3)*100 #, average='weighted'
    Evaluation_Results.iloc[i]['AUC-ROC score']=round(roc_auc_score(Test_Y, pred),3)*100 #, multi_class='ovr'
    print('{}{}\033[1m Evaluating {} \033[0m{}{}\n'.format('<'*3,'-'*35,Evaluation_Results.index[i], '-'*35,'>'*3))
    print('Accuracy = {}%'.format(round(accuracy_score(Test_Y, pred),3)*100))
    print('F1 Score = {}%'.format(round(f1_score(Test_Y, pred),3)*100)) #, average='weighted'
    print('\n \033[1mConfusiton Matrix:\033[0m\n',confusion_matrix(Test_Y, pred))
    print('\n\033[1mClassification Report:\033[0m\n',classification_report(Test_Y, pred))
    

#Visualising Function
def AUC_ROC_plot(Test_Y, pred):    
    ref = [0 for _ in range(len(Test_Y))]
    ref_auc = roc_auc_score(Test_Y, ref)
    lr_auc = roc_auc_score(Test_Y, pred)

    ns_fpr, ns_tpr, _ = roc_curve(Test_Y, ref)
    lr_fpr, lr_tpr, _ = roc_curve(Test_Y, pred)


task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()


# ---

# ## 1. Logistic Regression:

# In[12]:


# Building Logistic Regression Classifier

LR_model = LogisticRegression()
LR = LR_model.fit(Train_X, Train_Y)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()

pred = LR.predict(Test_X)
pred_prob = LR.predict_proba(Test_X)
Classification_Summary(pred,pred_prob,0)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()


# ---

# ## 2. Naive Bayes Classfier:

# In[ ]:


# Building Naive Bayes Classifier

NB_model = BernoulliNB()
NB = NB_model.fit(Train_X, Train_Y)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()

pred = NB.predict(Test_X)
pred_prob = NB.predict_proba(Test_X)
Classification_Summary(pred,pred_prob,3)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()


# ## 3. Decisoin Tree Classfier:

# In[13]:


# Building Decision Tree Classifier

DT_model = DecisionTreeClassifier()
DT = DT_model.fit(Train_X, Train_Y)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()

pred = DT.predict(Test_X)
pred_prob = DT.predict_proba(Test_X)
Classification_Summary(pred,pred_prob,1)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()


# ---

# ## 4. Random Forest Classfier:

# In[14]:


# Building Random Forest Classifier

RF_model = RandomForestClassifier()
RF = RF_model.fit(Train_X, Train_Y)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()

pred = RF.predict(Test_X)
pred_prob = RF.predict_proba(Test_X)
Classification_Summary(pred,pred_prob,2)

task_times.append(datetime.datetime.now() - start_time)
start_time = datetime.datetime.now()


# In[ ]:


# Save the time taken for each task

with open('time.txt', 'w') as f:
    for i in range(len(task_names)):
        f.write(f'{task_names[i]}: {task_times[i]}\n')


# ---
