#!/usr/bin/env python
# coding: utf-8

# # 1. 2차원 데이터의 정리

# ## 1.1. 두 데이터 사이의 관계를 나타내는 지표

# In[3]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('precision', '3')


# In[6]:


df = pd.read_csv('../static analysis/data/ch2_scores_em.csv',
                 index_col='student number')


# In[10]:


en_scores = np.array(df['english'])[:10]
ma_scores = np.array(df['mathematics'])[:10]

scores_df = pd.DataFrame({'english':en_scores,
                          'mathematics':ma_scores},
                        index=pd.Index(['A', 'B', 'C', 'D', 'E',
                                        'F', 'G', 'H', 'I', 'J'],
                                       name='student'))
scores_df


# ### 1.1.1. 共分散

# In[12]:


summary_df = scores_df.copy()
summary_df['english_deviation'] =\
    summary_df['english'] - summary_df['english'].mean()
summary_df['mathematics_deviation'] =\
    summary_df['mathematics'] - summary_df['mathematics'].mean()
summary_df['product of deviations'] =\
    summary_df['english_deviation'] * summary_df['mathematics_deviation']
summary_df


# In[15]:


summary_df['product of deviations'].mean()


# In[16]:


cov_mat = np.cov(en_scores, ma_scores, ddof=0)
cov_mat


# In[17]:


cov_mat[0, 0], cov_mat[1, 1]


# In[18]:


np.var(en_scores, ddof=0), np.var(ma_scores, ddof=0)


# In[19]:


cov_mat[0, 1], cov_mat[1, 0]


# In[20]:


np.cov(en_scores, ma_scores, ddof=0)[0, 1] /\
(np.std(en_scores) * np.std(ma_scores))


# In[21]:


np.corrcoef(en_scores, ma_scores)


# In[22]:


scores_df.corr()


# ## 1.2. 2차원의 데이터의 시각화

# ### 1.2.1. 산점도

# In[23]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


english_scores = np.array(df['english'])
math_scores = np.array(df['mathematics'])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
# 산점도
ax.scatter(english_scores, math_scores)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')

plt.show()


# ### 1.2.2. 회귀직선

# In[25]:


# 계수ß_0와 ß_1를 구한다
poly_fit = np.polyfit(english_scores, math_scores, 1)
# ß_0+ =ß_1 x를 반환하는 함수를 작성
poly_1d = np.poly1d(poly_fit)
#직선을 그리기 위해 x좌표를 생성
xs = np.linspace(english_scores.min(), english_scores.max())
#xs에 대응하는 y좌표를 구한다.
ys = poly_1d(xs)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.scatter(english_scores, math_scores, label='scores')
ax.plot(xs, ys, color='gray',
        label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
#범례의 표시
ax.legend(loc='upper left')

plt.show()


# ### 1.2.3. 히트맵

# In[28]:


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

c = ax.hist2d(english_scores, math_scores,
              bins=[9, 8], range=[(35, 80), (55, 95)])
ax.set_xlabel('english')
ax.set_ylabel('mathematics')
ax.set_xticks(c[1])
ax.set_yticks(c[2])
#컬러 바의 표시
fig.colorbar(c[3], ax=ax)
plt.show()


# ## 1.3. 앤스컴의 예

# In[29]:


# npy 형식으로 저장된 NumPy array를 읽어들인다
anscombe_data = np.load('../static analysis/data/ch3_anscombe.npy')
print(anscombe_data.shape)
anscombe_data[0]


# In[31]:


stats_df = pd.DataFrame(index=['X_mean', 'X_variance', 'Y_mean',
                               'Y_vareance', 'X&Y_correlation',
                               'X&Y_regression line'])
for i, data in enumerate(anscombe_data):
    dataX = data[:, 0]
    dataY = data[:, 1]
    poly_fit = np.polyfit(dataX, dataY, 1)
    stats_df[f'data{i+1}'] =\
        [f'{np.mean(dataX):.2f}',
         f'{np.var(dataX):.2f}',
         f'{np.mean(dataY):.2f}',
         f'{np.var(dataY):.2f}',
         f'{np.corrcoef(dataX, dataY)[0, 1]:.2f}',
         f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x']
stats_df


# In[32]:


# 그래프를 그리기 위한 영역을 2x2개 생성
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex=True, sharey=True)

xs = np.linspace(0, 30, 100)
for i, data in enumerate(anscombe_data):
    poly_fit = np.polyfit(data[:,0], data[:,1], 1)
    poly_1d = np.poly1d(poly_fit)
    ys = poly_1d(xs)
    #그리는 영역을 선택
    ax = axes[i//2, i%2]
    ax.set_xlim([4, 20])
    ax.set_ylim([3, 13])
    # 타이틀을 부여
    ax.set_title(f'data{i+1}')
    ax.scatter(data[:,0], data[:,1])
    ax.plot(xs, ys, color='gray')

# 그래프 사이의 간격을 좁힘
plt.tight_layout()
plt.show()


# In[ ]:




