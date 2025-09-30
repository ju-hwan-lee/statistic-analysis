#!/usr/bin/env python
# coding: utf-8

# # 1. 추측통계의 기본

# ## 1.1. 모집단과 표본

# ### 1.1.1. 표본의 추출방법

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('../static analysis/data/ch4_scores400.csv')
scores = np.array(df['score'])
scores[:10]


# In[3]:


np.random.choice([1, 2, 3], 3)


# In[4]:


np.random.choice([1, 2, 3], 3, replace=False)


# In[5]:


np.random.seed(0)
np.random.choice([1, 2, 3], 3)


# In[6]:


np.random.seed(0)
sample = np.random.choice(scores, 20)
sample.mean()


# In[7]:


scores.mean()


# In[8]:


for i in range(5):
    sample = np.random.choice(scores, 20)
    print(f'{i+1}번째 무작위 추출로 얻은 표본평균', sample.mean())


# In[9]:


dice = [1, 2, 3, 4, 5, 6]
prob = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]


# In[10]:


num_trial = 100
sample = np.random.choice(dice, num_trial, p=prob)
sample


# In[11]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(scores, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
plt.show()


# In[14]:


np.random.choice(scores)


# In[15]:


sample = np.random.choice(scores, 10000)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
plt.show()


# In[16]:


sample_means = [np.random.choice(scores, 20).mean()
                for _ in range(10000)]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample_means, bins=100, range=(0, 100), density=True)
# 모평균을 세로선으로 표시
ax.vlines(np.mean(scores), 0, 1, 'gray')
ax.set_xlim(50, 90)
ax.set_ylim(0, 0.13)
ax.set_xlabel('score')
ax.set_ylabel('relative frequency')
plt.show()

