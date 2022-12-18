#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,StackingClassifier 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load dataset
dataset= pd.read_excel("Output-table-prep.xlsx")


# In[3]:


dataset.head()


# In[4]:


dataset.iloc[1,:]


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# ###  here the raised and needed column should be changed to numeric datatype

# In[7]:


raised_amount  = []
for i in range(len(dataset['raised'])):
    if 'Rs' in dataset['raised'][i]:
        raised_amount.append(int(dataset['raised'][i][3:].replace(",","")))
    elif '$' in dataset['raised'][i]:
        raised_amount.append(int(dataset['raised'][i][1:].replace(",","")))


# In[8]:


dataset['raised_amount'] = raised_amount


# In[9]:


needed_amount = []
for i in range(len(dataset['needed'])):
    if '$' in dataset['needed'][i]:
        needed_amount.append(int(dataset['needed'][i][1:].replace(",","")))
    elif 'Rs' in dataset['needed'][i]:
        needed_amount.append(int(dataset['needed'][i][3:].replace(",","")))


# In[10]:


dataset['needed_amount'] = needed_amount


# In[11]:


dataset[['raised_amount','needed_amount']].dtypes


# In[12]:


#dropping raised,needed column after conversion
dataset.drop(['raised','needed'],axis=1,inplace=True)


# ### check null values in the dataset 

# In[13]:


dataset.isnull().sum()


# In[14]:


dataset['title'] = dataset['title'].fillna('unknown')


# # Exploratory Data Analysis 

# In[15]:


sns.countplot(data=dataset,x='genuine')


# #### We have nearly equal no of labels in predictor attribute. so mostly the predictor attribute is balanced

# In[16]:


sns.countplot(data=dataset,x='tax benifits',hue='genuine')
plt.legend()


# #### Here the fundraisers who don't have tax benifits are mostly not genuine 

# In[17]:


plt.figure(figsize=(8,4))
sns.scatterplot(y='raised_amount',x='supporters',data=dataset,hue='genuine')


# In[18]:


not_genuine = dataset.loc[(dataset['genuine']=='no')]
sns.scatterplot(y='needed_amount',x='supporters',data=not_genuine,hue='genuine')


# In[19]:


sns.scatterplot(y='raised_amount',x='supporters',data=not_genuine,hue='genuine')


# #### Here we see a linear relationship between no of supporters and raised_amount 
# #### Also can see that fundraiser who have gained people's trust is genuine 

# In[20]:


sns.distplot(dataset['raised_amount'],bins=30)


# In[21]:


sns.distplot(dataset['needed_amount'],bins=30)


# In[22]:


sns.distplot(dataset['supporters'],bins=30)


# # Categorical features
# 

# In[23]:


low_cardinality_cols = [col for col in dataset.columns if dataset[col].nunique()<10 and dataset[col].dtype=="object"]
print(low_cardinality_cols)


# In[24]:


high_cardinality_cols = [col for col in dataset.columns if dataset[col].nunique()>10 and dataset[col].dtype=="object"]
print(high_cardinality_cols)


# In[25]:


data = dataset.copy()


# In[26]:


data[low_cardinality_cols] = pd.get_dummies(dataset[low_cardinality_cols],drop_first="True")


# In[27]:


data


# ### Replace the state names with probabiltites w.r.t genuine column 

# In[28]:


by_state = pd.DataFrame(dataset.groupby(['State','genuine']).count())
by_state.reset_index(inplace=True)


# In[29]:


by_state = by_state.iloc[:,[0,1,2]]


# In[30]:


state_values = dict(by_state.groupby('State').City.sum())


# In[31]:


prob_by_state = [by_state['City'][i]/state_values[by_state['State'][i]] for i in range(len(by_state['State']))]   


# In[32]:


by_state['prob_by_state'] = prob_by_state


# In[33]:


by_state_no = by_state.loc[by_state['genuine']=='no']
by_state_yes = by_state.loc[by_state['genuine']=='yes']


# In[34]:


x1 = list(by_state_yes['State'])
y1 = list(by_state_yes['prob_by_state'])
x2 = list(by_state_no['State'])
y2 = list(by_state_no['prob_by_state']) 


# In[35]:


data_y = data.loc[data['genuine']==1]
data_y['State1'] = data_y['State'].replace(x1,y1)
data_y


# In[36]:


data_n = data.loc[data['genuine']==0]
data_n['State1'] = data_n['State'].replace(x2,y2)
data_n


# In[37]:


data = pd.concat([data_n,data_y])


# In[38]:


def state(string):
    try:
        return abs(float(by_state_no.loc[by_state_no['State']==string]['prob_by_state'])-float(by_state_yes.loc[by_state_yes['State']==string]['prob_by_state']))
    except:
        return 0
    
data['statediff'] = data.State.map(state)


# #### Tokenization — convert sentences to words
# #### Removing unnecessary punctuation, tags Removing stop words — frequent words such as ”the”, ”is”, etc. that do not have specific semantic
# #### Stemming — words are reduced to a root by removing inflection through dropping unnecessary characters, usually a suffix.
# #### Lemmatization — Another approach to remove inflection by determining the part of speech and utilizing detailed database of the language.

# In[39]:


import nltk
from nltk.tokenize import word_tokenize
titles = dataset.title.str.cat(sep=' ')


# In[40]:


tokens = word_tokenize(titles)
vocabulary = set(tokens)
print(len(vocabulary))


# In[41]:


frequency_dist = nltk.FreqDist(tokens)
repeated_words = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:75]


# In[42]:


repeated_words = [i.upper() for i in repeated_words]


# In[43]:


repeated_words_new = []

for i in range(len(repeated_words)):
    words = re.sub("[)!&,.''(]",'',repeated_words[i])
    repeated_words_new.append(words)
repeated_words = list(set(repeated_words_new))    


# In[44]:


repeated_words.sort()
repeated_words = repeated_words[1:]


# In[45]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequency_dist)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# ## Jaccard similairty

# In[46]:


def regex_string(column_name):
    new_column = []
    string = ''
    for i in range(len(data[column_name])):
        string = re.sub("[!&',.0-9]",'',data[column_name][i].upper())
        string = re.sub(data['benefited_to'][0].upper(),"",string)
        string = re.sub(data['posted_by'][0].upper(),"",string)
        new_column.append(string)
    return new_column    


# In[47]:


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return np.round((intersection/union),4)


# In[48]:


data['new_title'] = regex_string('title')


# In[49]:


jaccard_sim = []
for i in range(len(data['title'])):
    jaccard_sim.append(jaccard(repeated_words,data['new_title'][i].split(" ")))


# In[50]:


data['jaccard_title'] = jaccard_sim


# ### drop high cardinality cols

# In[51]:


data1 = data.copy()


# In[52]:


data1 = data.drop(high_cardinality_cols,axis=1)


# In[53]:


data1.drop('new_title',axis=1,inplace=True)


# ## Handling numerical columns 

# ### For raised_amount,needed_amount column

# In[54]:


data1['perc_collected'] =  data1['raised_amount']/data1['needed_amount'] 


# In[55]:


def remove_outliers(dataset,column_name):
    Q1 = np.quantile(dataset[column_name],0.25)
    Q3 = np.quantile(dataset[column_name],0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    higher= Q3+1.5*IQR
    not_outliers = list(dataset[column_name].loc[(dataset[column_name]>lower) & (dataset[column_name]<higher)].index)
    dataset_new = dataset.iloc[not_outliers,:]
    return dataset_new


# In[56]:


def make_quantiles(dataset,column_name):
    quantiles = list(np.quantile(dataset[column_name],[0,0.25,0.5,0.75,1]))
    return quantiles


# In[57]:


def make_slab(data):
    if data>=quantiles[0] and data<quantiles[1]:
        return 1
    elif data>=quantiles[1] and data<quantiles[2]:
        return 2
    elif data>=quantiles[2] and data<quantiles[3]:
        return 3
    elif data>=quantiles[3] and data<quantiles[4]:
        return 4
    else:
        return 5


# ### supporters_slab

# In[58]:


data_new = remove_outliers(data1,'supporters')
quantiles = make_quantiles(data_new,'supporters')
supporters_slab = data1.supporters.map(make_slab)
data1['supporters_slab'] = supporters_slab


# In[59]:


quantiles


# ### raised_slab

# In[60]:


data_new = remove_outliers(data1,'raised_amount')
quantiles = make_quantiles(data_new,'raised_amount')
raised_slab =data1.raised_amount.map(make_slab)
data1['raised_slab'] = raised_slab


# In[61]:


quantiles


# ### needed_slab

# In[62]:


data_new = remove_outliers(data1,'needed_amount')
quantiles = make_quantiles(data_new,'needed_amount')
needed_slab = data1.needed_amount.map(make_slab)
data1['needed_slab'] = needed_slab


# In[63]:


quantiles


# ### Split the data 

# In[64]:


data1['tax benifits'] = data1['tax benifits'].astype('int64')
data1['genuine'] = data1['genuine'].astype('int64')


# In[65]:


X = data1.drop(['genuine'],axis=1)
y = data1['genuine']


# In[66]:


print(X.shape,y.shape)


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)


# In[68]:


X_train


# ## Hyperparamter tuning using Randomsearchcv  

# #### for Decisiontree

# In[69]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
preds = dt.predict(X_test)
print(classification_report(y_test,preds))


# In[70]:


DecisionTreeClassifier()


max_depth = [int(x) for x in np.linspace(10, 2000,10)]

min_samples_split = [int(x) for x in np.linspace(2,100,1)]

min_samples_leaf = [int(x) for x in np.linspace(1,100,1)]

max_leaf_nodes = [int(x) for x in np.linspace(10, 2000,10)]

ccp_alpha10         = [np.round(random.random()/10,4) for x in range(1,100)]
ccp_alpha100        = [np.round(random.random()/100,4) for x in range(1,100)]
ccp_alpha1000       = [np.round(random.random()/1000,4) for x in range(1,100)]

ccp_alpha           = ccp_alpha10+ccp_alpha100+ccp_alpha1000


random_grid = {'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini'],
              'ccp_alpha':ccp_alpha,
              'max_leaf_nodes':max_leaf_nodes}


# In[71]:


dt1 = DecisionTreeClassifier()
dt_randomcv = RandomizedSearchCV(estimator=dt1,param_distributions=random_grid,n_jobs=-1)
dt_randomcv.fit(X_train,y_train)


# In[72]:


params_dt = dt_randomcv.best_params_
params_dt


# In[73]:


dt = DecisionTreeClassifier(min_samples_leaf=params_dt['min_samples_leaf'],
                            min_samples_split=params_dt['min_samples_split'],
                           max_leaf_nodes=params_dt['max_leaf_nodes'],
                           max_features=len(X_train.columns),
                           max_depth=params_dt['max_depth'],
                           criterion=params_dt['criterion'],
                           ccp_alpha=params_dt['ccp_alpha'])
dt.fit(X_train,y_train)
preds = dt.predict(X_test)
print(classification_report(y_test,preds))


# #### for RandomForest

# In[74]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
preds = rf.predict(X_test)
print(classification_report(y_test,preds))


# In[75]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]


max_depth = [int(x) for x in np.linspace(10, 2000,10)]

min_samples_split = [int(x) for x in np.linspace(2,100,1)]

min_samples_leaf = [int(x) for x in np.linspace(1,100,1)]

ccp_alpha10         = [np.round(random.random()/10,4) for x in range(1,100)]
ccp_alpha100        = [np.round(random.random()/100,4) for x in range(1,100)]
ccp_alpha1000       = [np.round(random.random()/1000,4) for x in range(1,100)]

ccp_alpha           = ccp_alpha10+ccp_alpha100+ccp_alpha1000

class_weight     = ['balanced','balanced_subsample']

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini'],
              'ccp_alpha':ccp_alpha,
              'class_weight':class_weight}


# In[76]:


rf1 = RandomForestClassifier()
rf_randomcv = RandomizedSearchCV(estimator=rf1,param_distributions=random_grid,n_jobs=-1)
rf_randomcv.fit(X_train,y_train)


# In[77]:


params = rf_randomcv.best_params_


# In[78]:


rf = RandomForestClassifier(n_estimators=params['n_estimators'],min_samples_split= params['min_samples_split'],
                            min_samples_leaf= params['min_samples_leaf'],
                            max_features=len(X_train.columns),
                            criterion=params['criterion'],max_depth=params['max_depth'],
                           class_weight=params['class_weight'],ccp_alpha=params['ccp_alpha'])
rf.fit(X_train,y_train)
preds = rf.predict(X_test)
print(classification_report(y_test,preds))


# #### for adaboost

# In[79]:


adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
adb.fit(X_train,y_train)
preds = adb.predict(X_test)
print(classification_report(y_test,preds))


# In[80]:


alpha10         = [np.round(random.random()/10,4) for x in range(1,100)]
alpha100        = [np.round(random.random()/100,4) for x in range(1,100)]
alpha1000       = [np.round(random.random()/1000,4) for x in range(1,100)]

learning_rate = alpha10+alpha100+alpha1000
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]

random_grid = {'learning_rate':learning_rate,
              'n_estimators':n_estimators,
              'algorithm':['SAMME','SAMME.R']}


# In[81]:


adb1 = AdaBoostClassifier()
ada_randomcv = RandomizedSearchCV(estimator=adb1,param_distributions=random_grid,n_jobs=-1)


# In[82]:


ada_randomcv.fit(X_train,y_train)


# In[83]:


params_adb = ada_randomcv.best_params_
params_adb


# In[84]:


adbr = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=params_dt['min_samples_leaf'],
                            min_samples_split=params_dt['min_samples_split'],
                           max_leaf_nodes=params_dt['max_leaf_nodes'],
                           max_features=len(X_train.columns),
                           max_depth=params_dt['max_depth'],
                           criterion=params_dt['criterion'],
                           ccp_alpha=params_dt['ccp_alpha'])
                          ,n_estimators=params_adb['n_estimators'],
                           learning_rate=params_adb['learning_rate'],
                           algorithm=params_adb['algorithm'])
adbr.fit(X_train,y_train)
preds = adbr.predict(X_test)
print(classification_report(y_test,preds))


# In[85]:


level0 = [('dt_classifier',DecisionTreeClassifier(min_samples_leaf=params_dt['min_samples_leaf'],
                            min_samples_split=params_dt['min_samples_split'],
                           max_leaf_nodes=params_dt['max_leaf_nodes'],
                           max_features=len(X_train.columns),
                           max_depth=params_dt['max_depth'],
                           criterion=params_dt['criterion'],
                           ccp_alpha=params_dt['ccp_alpha'])),
          ('rf_classifier', RandomForestClassifier(n_estimators=params['n_estimators'],
                                                   min_samples_split= params['min_samples_split'],
                                                   min_samples_leaf= params['min_samples_leaf'],
                                                   max_features=len(X_train.columns),
                                                   criterion=params['criterion'],
                                                   max_depth=params['max_depth'],
                                                   class_weight=params['class_weight'],
                                                   ccp_alpha=params['ccp_alpha'])),
          ('ad_classifier',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
              min_samples_leaf=params_dt['min_samples_leaf'],
                            min_samples_split=params_dt['min_samples_split'],
                           max_leaf_nodes=params_dt['max_leaf_nodes'],
                           max_features=len(X_train.columns),
                           max_depth=params_dt['max_depth'],
                           criterion=params_dt['criterion'],
                           ccp_alpha=params_dt['ccp_alpha'])
                          ,n_estimators=params_adb['n_estimators'],
                           learning_rate=params_adb['learning_rate'],
                           algorithm=params_adb['algorithm']))]

level1 = LogisticRegression()


model = StackingClassifier(estimators=level0,final_estimator=level1,n_jobs=-1)


# In[86]:


model.fit(X_train,y_train)
y_preds = model.predict(X_test)


# In[87]:


print(confusion_matrix(y_test,y_preds))
print(accuracy_score(y_test,y_preds))


# In[88]:


print(classification_report(y_test,y_preds))


# In[89]:


print(X_test.loc[849,:])
print(y_test.loc[849])


# In[90]:


x = [3,0,600,500000,0.581699,0.16339,0.065600,0.001200,1,1,2]


# In[91]:


x = np.reshape(x,(1,-1))
#model.predict(x)


# In[92]:


filename = 'modellib.sav'


# In[93]:


joblib.dump(model, filename)
 


# In[94]:


loaded_model = joblib.load(filename)


# In[95]:


preds = loaded_model.predict_proba(x)


# In[96]:


preds[0]


# In[97]:


x1 = loaded_model.predict_proba(x)


# In[98]:


x1


# In[99]:


x1.tolist()[0][0]

