#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Spark Foundation
# ( Data Science and Business Analytics Intern )
#  # Aim: To predict the score of a student when he/she studies for 9.25 hours.
# # Prepared by :Rahul singh

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[2]:


dataURL='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'


# In[3]:


df=pd.read_csv(dataURL)


# In[4]:


df.head()


# In[6]:


df.plot(x='Hours',y='Scores',style='x')
plt.title('Hours vs Percentage')
plt.xlabel('No. of hours')
plt.ylabel('Percentage')
plt.show()


# In[7]:


x=df.iloc[:,0:1]
y=df.iloc[:,1:]


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[9]:


lr=LinearRegression()


# In[10]:


lr.fit(x_train,y_train)


# In[11]:


lr.score(x_train,y_train)


# In[12]:


lr.score(x_test,y_test)


# In[13]:


pred=lr.predict(x_test)


# In[14]:


print(mean_squared_error(pred,y_test))


# In[15]:


print(np.sqrt(mean_squared_error(pred,y_test)))


# In[16]:


line = lr.coef_*x + lr.intercept_

plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[17]:


df2=pd.DataFrame(y_test)
df2


# In[19]:


df2['Predicted values']=pred


# In[20]:


df2


# In[21]:


hours= [[9.25]]


# In[23]:


prediction=lr.predict(hours)


# In[24]:


prediction


# In[ ]:




