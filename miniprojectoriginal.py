#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
# In[2]:


data = pd.read_csv('C:\\Users\\hp\\OneDrive\\Documents\\onlinedataset.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data.type.value_counts()


# In[7]:


type=data['type'].value_counts()


# In[8]:


transaction=type.index
quantity = type.values


# In[9]:


import plotly.express as px
figure = px.pie(data,values= quantity,names=transaction,hole=0.5,title="Distribution of Transaction Type")
figure.show()


# In[10]:


# Select numeric data
numeric_data = data.select_dtypes(include=['number'])

# Compute correlation
correlation = numeric_data.corr()

# Sort and display correlations with 'isFraud'
if 'isFraud' in correlation:
    print(correlation['isFraud'].sort_values(ascending=False))
else:
    print("'isFraud' column not found in the dataset!")


# In[11]:


data['type'] = data['type'].map({'CASH_OUT:':1,'PAYMENT':2,'CASH_IN':3,'TRANSFER':4,'DEBIT':5})


# In[12]:


x = np.array(data[['type','amount','oldbalanceOrg','newbalanceOrig']])
y = np.array(data[['isFraud']])


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
xtrain,xtest,ytrain,ytest =train_test_split(x,y,test_size=0.2,random_state=42)
xtrain.shape


# In[14]:


model = DecisionTreeClassifier()
from sklearn.impute import SimpleImputer

# Impute missing values with the mean of the columns
imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)
model.fit(xtrain, ytrain)
model.score(xtest,ytest)


# In[15]:


feature = np.array([[4,9020.60,9000.60,0.0]])
model.predict(feature)


# In[ ]:




