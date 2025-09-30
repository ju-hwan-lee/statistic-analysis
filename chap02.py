#!/usr/bin/env python
# coding: utf-8

# # 1.1차원 데이터의 정리

# ## 1.1. 데이터 중심의 지표

# In[76]:


import numpy as np
import pandas as pd

#Jupyter Notebook의 출력을 소수점 이하 3자리로 제한
get_ipython().run_line_magic('precision', '3')


# In[77]:


df = pd.read_csv('../data/ch2_scores_em.csv', 
                 index_col='student number')
#df의 처음 5행을 표시
df.head()


# In[78]:


scores = np.array(df['english'])[:10]
scores


# In[79]:


scores_df = pd.DataFrame({'score':scores},
                         index=pd.Index(['A', 'B', 'C', 'D', 'E',
                                         'F', 'G', 'H', 'I', 'J'],
                                        name='student'))
scores_df


# ## 1.1.1. 평균값

# In[80]:


sum(scores) / len(scores)


# In[81]:


np.mean(scores)


# In[82]:


scores_df.mean()


# ## 1.1.2. 중앙값

# In[83]:


sorted_scores = np.sort(scores)
sorted_scores


# In[84]:


n = len(sorted_scores)
if n % 2 == 0:
    m0 = sorted_scores[n//2-1]
    m1 = sorted_scores[n//2]
    median = (m0 + m1) / 2
else:
    median = sorted_scores[(n+1)//2-1]
median


# In[85]:


np.median(scores)


# In[86]:


scores_df.median()


# ## 1.1.3. 최빈값

# In[87]:


pd.Series([1, 1, 1, 2, 2, 3]).mode()


# In[88]:


pd.Series([1, 2, 3, 4, 5]).mode()


# # 1.2. 데이터의 산포도 지표

# ## 1.2.1. 분산과 표준편차

# ### 1.2.1.1. 편차

# In[89]:


mean = np.mean(scores)
deviation = scores - mean
deviation


# In[90]:


another_scores = [50,  60, 58, 54, 51, 56, 57, 53, 52, 59]
another_mean = np.mean(another_scores)
another_deviation = another_scores - another_mean
another_deviation


# In[91]:


np.mean(deviation)


# In[92]:


np.mean(another_deviation)


# In[93]:


summary_df = scores_df.copy()
summary_df['deviation'] = deviation
summary_df


# In[94]:


summary_df.mean()


# ### 1.2.1.2. 분산

# In[95]:


np.mean(deviation ** 2)


# In[96]:


np.var(scores)


# In[97]:


scores_df.var


# In[98]:


summary_df['square of deviation'] = np.square(deviation)
summary_df


# In[99]:


summary_df.mean()


# ### 1.2.1.3. 표준편차

# In[100]:


np.sqrt(np.var(scores, ddof=0))


# In[101]:


np.std(scores, ddof=0)


# ## 1.2.2. 범위와 4분위수 범위

# ### 1.2.2.1. 범위

# In[102]:


np.max(scores) - np.min(scores)


# In[103]:


scores_Q1 = np.percentile(scores, 25)
scores_Q3 = np.percentile(scores, 75)
scores_IQR = scores_Q3 - scores_Q1
scores_IQR


# ## 1.2.3. 데이터의 지표 정리

# In[104]:


pd.Series(scores).describe()


# In[105]:


z = (scores - np.mean(scores)) / np.std(scores)
z


# In[106]:


np.mean(z), np.std(z, ddof=0)


# In[107]:


z = 50 + 10 * (scores - np.mean(scores)) / np.std(scores)
z


# In[108]:


scores_df['deviation value'] = z
scores_df


# # 1.4. 데이터의 시각화

# In[109]:


#50명의 영어 점수 array
english_scores = np.array(df['english'])
#Series로 변환하여 describe를 표시
pd.Series(english_scores).describe()


# ## 1.4.1. 도수분포표

# In[110]:


freq, _ = np.histogram(english_scores, bins=10, range=(0, 100))
freq


# In[111]:


# 0~10, 10~20, ... 이라는 문자열의 리스트를 작성
freq_class = [f' {i}~{i+10}' for i in range(0, 100, 10)]
#freq_class를 인덱스로 DataFrame을 작성
freq_dist_df = pd.DataFrame({'frequency':freq},
                            index=pd.Index(freq_class,
                                           name='class'))
freq_dist_df


# In[112]:


for a in range(7):
    print (a)


# In[113]:


for a in range(10, 5, -1):
    print(a)


# In[114]:


for a in range(20, 31, 2):
    print(a)


# In[115]:


total = 0
for i in range(1, 10):
    total = total + i
print(total)


# In[116]:


total=0
for i in range(1, 10, 2):
    total = total + i
print(total)


# In[117]:


class_value = [(i+(i+10))//2 for i in range(0, 100, 10)]
class_value


# In[118]:


rel_freq = freq / freq.sum()
rel_freq


# In[119]:


cum_rel_freq = np.cumsum(rel_freq)
cum_rel_freq


# In[120]:


freq_dist_df['class value'] = class_value
freq_dist_df['relative frequency'] = rel_freq
freq_dist_df['cumulative relative frequency'] = cum_rel_freq
freq_dist_df = freq_dist_df[['class value', 'frequency',
                             'relative frequency', 'cumulative relative frequency']]
freq_dist_df


# In[121]:


freq_dist_df.loc[freq_dist_df['frequency'].idxmax(), 'class value']


# ## 1.4.2. 히스토그램

# In[122]:


# Matplotlib의 pyplot 모듈을 plt라는 이름으로 임포트
import matplotlib.pyplot as plt

# 그래프가 Jupyter notebook 위에 표시
get_ipython().run_line_magic('matplotlib', 'inline')


# In[128]:


#캔버스를 생성
#figsize로 가로•세로 크기를 지정
fig = plt.figure(figsize=(10, 6))
#캔버스 위에 그래프를 그리기 위한 영역을 지정
#인수는 영역을 1x1개 지정. 하나의 영역에 그린다는 것을 의미
ax = fig.add_subplot(111)

#계급수를 110으로 하여 히스토그램을 그림
freq, _, _ = ax.hist(english_scores, bins=10, range=(0, 100))
#X축에 레이블 부여
ax.set_xlabel('scores')
#Y축에 레이블 부여
ax.set_ylabel('person number')
#X축을 0, 10, 20, ...,100 눈금으로 구분
ax.set_xticks(np.linspace(0, 100, 10+1))
#Y축을 0, 1, 2, ...의 눈금으로 구분
ax.set_yticks(np.arange(0, freq.max()+1))
#그래프 표시
plt.show()


# In[129]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

freq, _ , _ = ax.hist(english_scores, bins=25, range=(0, 100))
ax.set_xlabel('scores')
ax.set_ylabel('person number')
ax.set_xticks(np.linspace(0, 100, 25+1))
ax.set_yticks(np.arange(0, freq.max()+1))
plt.show()


# In[130]:


fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
#Y축의 스케일이 다른 그래프를 ax1과 동일한 영역에 생성
ax2 = ax1. twinx()

#상대도수의 히스토그램으로 하기 위해서는, 도수를 데이터의 수로 나눌 필요가 있음
#이것은 hist의 인수 weight를 지정하면 실현가능
weights = np.ones_like(english_scores) / len(english_scores)
rel_freq, _, _ = ax1.hist(english_scores, bins=25,
                          range=(0, 100), weights=weights)

cum_rel_freq = np.cumsum(rel_freq)
class_value = [i+(i+4)//2 for i in range(0, 100, 4)]
#꺾은선 그래프로 그림
#인수 ls를 '--'로 하면 접선이 그려짐
#인수 marker를 'o'으로 하면 데이터 점을 그림
#인수 color를 'gray'로 하면 회색으로 지정
ax2.plot(class_value, cum_rel_freq,
         ls='--', marker='o', color='gray')
#꺾은선 그래프의 눈금선을 제거
ax2.grid(visible=False)

ax1.set_xlabel('scores')
ax1.set_ylabel('relative frequency')
ax2.set_ylabel('cumulative relative frequency')
ax1.set_xticks(np.linspace(0, 100, 25+1))

plt.show()


# In[131]:


fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111)
ax.boxplot(english_scores, tick_labels=['english'])

plt.show()


# In[134]:


import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
x=range(0,100)
y=[v*v for v in x]
ax1.plot(x,y)
ax2.bar(x,y)
plt.show()


# In[ ]:




