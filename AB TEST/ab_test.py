#!/usr/bin/env python
# coding: utf-8

# ## 分析A/B测试结果
# 
# 这个项目可以帮你确认你已经掌握了统计课程中涵盖的所有内容。 希望这个项目尽可能地涵盖所有内容。 祝你好运！
# 
# ## 目录
# - [简介](#intro)
# - [I - 概率](#probability)
# - [II - A/B 测试](#ab_test)
# - [III - 回归](#regression)
# 
# 
# <a id='intro'></a>
# ### 简介
# 
# 通常情况下，A/B 测试由数据分析师和数据科学家来完成。如果你在一些实践工作中遇到过这方面的问题，那学习起来就会更加游刃有余。
# 
# 对于这个项目，你将要了解的是电子商务网站运行的 A/B 测试的结果。你的目标是通过这个 notebook 来帮助公司弄清楚他们是否应该使用新的页面，保留旧的页面，或者应该将测试时间延长，之后再做出决定。
# 
# **使用该 notebook 的时候，请同步学习课堂内容，并回答与每个问题相关的对应测试题目。** 每个课堂概念的标签对应每个题目。这样可以确保你在完成项目的过程中的方法正确，并且你最终提交的内容会更加符合标准，不必担心出现错误。最后检查的时候，请确保你的提交内容符合 [审阅标准](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric) 中的所有标准。
# 
# <a id='probability'></a>
# #### I - 概率
# 
# 让我们先导入库，然后开始你的任务吧。

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` 现在，导入 `ab_data.csv` 数据，并将其存储在 `df` 中。  **使用你的 dataframe 来回答课堂测试 1 中的问题。**
# 
# a. 导入数据集，并在这里查看前几行：

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. 使用下面的单元格来查找数据集中的行数。

# In[3]:


df.shape[0]


# c. 数据集中独立用户的数量。

# In[4]:


df.user_id.nunique()


# d. 用户转化的比例。

# In[5]:


converted_ratio = df[df['converted'] == 1].user_id.nunique()/df.user_id.nunique()
converted_ratio


# e.  `new_page` 与 `treatment` 不一致的次数。

# In[6]:


treatment = df['group'] == 'treatment'#转换成布尔值
new_page = df['landing_page'] == 'new_page'
mismatch = treatment != new_page
mismatch.sum()


# 
# f. 是否有任何行存在缺失值？

# In[7]:


df.isnull().sum()


# 不存在缺失值

# `2.` 对于 **treatment** 不与 **new_page** 一致的行或 **control** 不与 **old_page** 一致的行，我们不能确定该行是否真正接收到了新的或旧的页面。我们应该如何处理这些行？在课堂中的 **测试 2** 中，给出你的答案。  
# 
# a. 现在，使用测试题的答案创建一个符合测试规格要求的新数据集。将新 dataframe 存储在 **df2** 中。

# In[8]:


df2 = df[~mismatch].copy()
df2.head()


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` 使用 **df2** 与下面的单元格来回答课堂中的 **测试3** 。
# 
# a.  **df2** 中有多少唯一的 **user_id**?

# In[10]:


df2.user_id.nunique()


# b.  **df2** 中有一个重复的 **user_id** 。它是什么？ 

# In[11]:


df2.user_id.duplicated().sum()


# c. 这个重复的  **user_id** 的行信息是什么？

# In[12]:


df2[df2.user_id.duplicated() == True]


# d. 删除 **一个** 含有重复的 **user_id** 的行， 但需要确保你的 dataframe 为 **df2**。

# In[13]:


df2.drop_duplicates('user_id',inplace=True)


# `4.` 在下面的单元格中，使用 **df2** 来回答与课堂中的 **测试 4** 相关的测试题目。
# 
# a. 不管它们收到什么页面，单个用户的转化率是多少？
# 

# In[14]:


df2.converted.mean()


# b. 假定一个用户处于 `control` 组中，他的转化率是多少？

# In[15]:


df2.query('group == "control"').converted.mean()


# c. 假定一个用户处于 `treatment` 组中，他的转化率是多少？
# 

# In[16]:


df2.query('group == "treatment"').converted.mean()


# d. 一个用户收到新页面的概率是多少？
# 

# In[17]:


df2.query('landing_page == "new_page"').shape[0]/df2.shape[0]


# e. 使用这个问题的前两部分的结果，给出你的建议：你是否认为有证据表明一个页面可以带来更多的转化？在下面写出你的答案。
# 
# **对照组的转换率比实验组的转换率高，由此可见，对照组的效果稍好于实验组，同时两组的转换率仅相差0.0016，所以不能完全断定，还需要更长的时间来判断。**

# <a id='ab_test'></a>
# ### II - A/B 测试
# 
# 请注意，由于与每个事件相关的时间戳，你可以在进行每次观察时连续运行假设检验。  
# 
# 然而，问题的难点在于，一个页面被认为比另一页页面的效果好得多的时候你就要停止检验吗？还是需要在一定时间内持续发生？你需要将检验运行多长时间来决定哪个页面比另一个页面更好？
# 
# 一般情况下，这些问题是A / B测试中最难的部分。如果你对下面提到的一些知识点比较生疏，请先回顾课程中的“描述统计学”部分的内容。

# `1.` 现在，你要考虑的是，你需要根据提供的所有数据做出决定。如果你想假定旧的页面效果更好，除非新的页面在类型I错误率为5％的情况下才能证明效果更好，那么，你的零假设和备择假设是什么？ 你可以根据单词或旧页面与新页面的转化率 **$p_{old}$** 与 **$p_{new}$** 来陈述你的假设。
# 
# **null hypothesis: 
# $p_{new}$ - $p_{old}$ <= 0**
# 
# **Alternative Hypothesis: 
# $p_{new}$ - $p_{old}$ > 0**

# In[18]:


p = df2.converted.mean()
p


# `2.` 假定在零假设中，不管是新页面还是旧页面， $p_{new}$ and $p_{old}$ 都具有等于 **转化** 成功率的“真”成功率，也就是说，  $p_{new}$ 与 $p_{old}$ 是相等的。此外，假设它们都等于**ab_data.csv** 中的 **转化** 率，新旧页面都是如此。  <br><br>
# 
# 每个页面的样本大小要与 **ab_data.csv** 中的页面大小相同。  <br><br>
# 
# 执行两次页面之间 **转化** 差异的抽样分布，计算零假设中10000次迭代计算的估计值。  <br><br>
# 
# 使用下面的单元格提供这个模拟的必要内容。如果现在还没有完整的意义，不要担心，你将通过下面的问题来解决这个问题。你可以通过做课堂中的 **测试 5** 来确认你掌握了这部分内容。<br><br>
# 
# a. 在零假设中，$p_{new}$ 的 **convert rate（转化率）** 是多少？
# 

# In[19]:


p_new = df2.query('landing_page == "new_page"').converted.mean() 
p_new


# b. 在零假设中， $p_{old}$  的 **convert rate（转化率）** 是多少？ <br><br>

# In[20]:


p_old = df2.query('landing_page == "old_page"').converted.mean() 
p_old


# c.  $n_{new}$ 是多少？

# In[21]:


new_page = df2.query('landing_page == "new_page"')
n_new = new_page.shape[0]
n_new


# d.  $n_{old}$?是多少？

# In[22]:


old_page = df2.query('landing_page == "old_page"')
n_old = old_page.shape[0]
n_old


# e. 在零假设中，使用 $p_{new}$ 转化率模拟 $n_{new}$ 交易，并将这些 $n_{new}$ 1's 与 0's 存储在 **new_page_converted** 中。(提示：可以使用  [numpy.random.choice](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html)。)

# In[23]:


new_page_converted = np.random.choice([1,0],size=n_new,replace=True,p=[p_new,1-p_new])
new_page_converted


# f. 在零假设中，使用 $p_{old}$ 转化率模拟 $n_{old}$ 交易，并将这些  $n_{old}$ 1's 与 0's 存储在 **old_page_converted** 中。

# In[24]:


old_page_converted = np.random.choice([1,0],size=n_old,replace=True,p=[p_old,1-p_old])


# g. 在 (e) 与 (f)中找到 $p_{new}$ - $p_{old}$ 模拟值。

# In[25]:


sample_diff = new_page_converted.mean()-old_page_converted.mean()
sample_diff


# h. 使用**a. 到 g. ** 中的计算方法来模拟 10,000个 $p_{new}$ - $p_{old}$ 值，并将这 10,000 个值存储在 **p_diffs** 中。

# In[26]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.choice([1,0],size=n_new,replace=True,p=[p_new,1-p_new])
    old_page_converted = np.random.choice([1,0],size=n_old,replace=True,p=[p_old,1-p_old])
    diff = new_page_converted.mean()-old_page_converted.mean()
    p_diffs.append(diff)
p_diffs = np.array(p_diffs)


# i. 绘制一个 **p_diffs** 直方图。这个直方图看起来像你所期望的吗？通过回答课堂上的匹配问题，确保你完全理解这里计算出的内容。

# In[27]:


plt.hist(p_diffs)


# j.  在**p_diffs**列表的数值中，有多大比例大于 **ab_data.csv** 中观察到的实际差值？

# In[28]:


act_diff = p_new - p_old
p_val = (act_diff < p_diffs).mean()
p_val


# k. 用文字解释一下你刚才在 **j.**中计算出来的结果。在科学研究中，这个值是什么？ 根据这个数值，新旧页面的转化率是否有区别呢？
# 
# **p_val=0.494 该p值大于一类错误阈值0.05，说明被视为极端值得比例足够大，所以不拒绝零假设，即旧页面的转化率优于新页面**
# 

# l. 我们也可以使用一个内置程序 （built-in）来实现类似的结果。尽管使用内置程序可能更易于编写代码，但上面的内容是对正确思考统计显著性至关重要的思想的一个预排。填写下面的内容来计算每个页面的转化次数，以及每个页面的访问人数。使用 `n_old` 与 `n_new` 分别引证与旧页面和新页面关联的行数。

# In[29]:


import statsmodels.api as sm

convert_old = sum(old_page.converted)
convert_new = sum(new_page.converted)
n_old = len(new_page)
n_new = len(old_page)


# m. 现在使用 `stats.proportions_ztest` 来计算你的检验统计量与 p-值。[这里](http://knowledgetack.com/python/statsmodels/proportions_ztest/) 是使用内置程序的一个有用链接。

# In[30]:


z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old],[n_new, n_old],alternative='larger')
z_score, p_value


# In[31]:


from scipy.stats import norm
norm.cdf(z_score),norm.ppf(1-0.05)#累计分布函数和分位点函数


# n. 根据上题算出的 z-score 和 p-value，我们认为新旧页面的转化率是否有区别？它们与 **j.** 与 **k.** 中的结果一致吗？
# 
# **z_score没有超过95%的置信度临界值1.6448536269514722，所以不拒绝零假设，旧页面优于新页面，这与之前的结论相同**

# <a id='regression'></a>
# ### III - 回归分析法之一
# 
# `1.` 在最后一部分中，你会看到，你在之前的A / B测试中获得的结果也可以通过执行回归来获取。<br><br>
# 
# a. 既然每行的值是转化或不转化，那么在这种情况下，我们应该执行哪种类型的回归？
# 
# **逻辑回归**

# b. 目标是使用 **statsmodels** 来拟合你在 **a.** 中指定的回归模型，以查看用户收到的不同页面是否存在显著的转化差异。但是，首先，你需要为这个截距创建一个列（ 原文：column） ，并为每个用户收到的页面创建一个虚拟变量列。添加一个 **截距** 列，一个 **ab_page** 列，当用户接收 **treatment** 时为1， **control** 时为0。

# In[32]:


df2[['control','ab_page']] = pd.get_dummies(df['group'])
df2.drop('control',axis=1,inplace=True)
df2.head()


# 
# c. 使用 **statsmodels** 导入你的回归模型。 实例化该模型，并使用你在 **b.** 中创建的2个列来拟合该模型，用来预测一个用户是否会发生转化。

# In[33]:


df2['intercept'] = 1
lm = sm.Logit(df2['converted'], df2[['intercept','ab_page']])
results = lm.fit()
results.summary()


# d. 请在下方提供你的模型摘要，并根据需要使用它来回答下面的问题。

# In[34]:


results.params


# e. 与 **ab_page** 关联的 p-值是多少？ 为什么它与你在 **II** 中发现的结果不同？<br><br>  **提示**: 与你的回归模型相关的零假设与备择假设分别是什么？它们如何与 **Part II** 中的零假设和备择假设做比较？
# 
# **该模型中 与 ab_page 关联的p值为0.190，逻辑回归中假设为：
# H0:Pnew = Pold
# H1:Pnew≠Pold
# 逻辑回归的零假设和备择假设与1部分的假设不同，零假设为旧页面和新页面的转换率相同，即自变量ab_page与反应变量converged可能无影响作用，备择假设为两者转换率不同，与零假设相反。之前计算的p值是用于单边检验的，而这个是双边检验，因此p值是有区别的，此处的p-值是指ab_page对应转换率的影响程度，p-值越小越具有显著性差异。
# 需要判断是新旧页面没差异还是旧页面的转换率大于新页面，需要进一步分析，由于变量之间无线性关系可以考虑加入高阶项或者交叉项变量。**

# 
# 
# f. 现在，你一定在考虑其他可能影响用户是否发生转化的因素。讨论为什么考虑将其他因素添加到回归模型中是一个不错的主意。在回归模型中添加附加项有什么弊端吗？
# 
# **由于特性太少不利于准确的模型的建立，可以考虑加入高阶项或者交叉项变量，同时，加入高阶项或者交叉项变量也有潜在的缺点:一般来说，如果在数据太少的情况下添加了太多的术语，我们可能无法得到收敛。但在这种情况下，由于我们有很多数据，收敛可能不是问题。另外增加额外的项也使系数的解释变得困难。**

# In[35]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()


# g. 现在，除了测试不同页面的转化率是否会发生变化之外，还要根据用户居住的国家或地区添加一个 effect 项。你需要导入 **countries.csv** 数据集，并将数据集合并在适当的行上。 [这里](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) 是链接表格的文档。 
# 
# 这个国家项对转化有影响吗？不要忘记为这些国家的列创建虚拟变量—— **提示: 你将需要为这三个虚拟变量增加两列。** 提供统计输出，并书面回答这个问题。

# In[36]:


import statsmodels.api as sm
df_new[['CA','US']]=pd.get_dummies(df_new['country'])[['CA',"US"]]
df_new.drop('country',axis=1,inplace=True)
df_new['intercept']=1
Xvars=[ 'intercept','CA','US','ab_page']
lm = sm.Logit (df_new['converted'],df_new[Xvars ] ) 
results = lm.fit()
results.summary()


# 基线为UK，在alpha为0.05时，每个解释变量的p值都大于0.05，说明解释变量与反应变量之间没有显著性影响

# In[37]:


### Model to look at interaction between page and country
df_new['intercept']=1
Xvars=[ 'intercept','CA','US']
lm = sm.Logit (df_new['ab_page'],df_new[Xvars ] ) 
results = lm.fit()
results.summary()


# 这些国家的p值更大，表明它们不是很有用(没有统计学意义)。

# h. 虽然你现在已经查看了国家与页面在转化率上的个体性因素，但现在我们要查看页面与国家/地区之间的相互作用，测试其是否会对转化产生重大影响。创建必要的附加列，并拟合一个新的模型。  
# 
# 提供你的摘要结果，以及根据结果得出的结论。
# 
# **提示：页面与国家/地区的相互作用**
# ```
# df3['new_CA'] = df3['new_page'] * df3['CA']
# df3['new_UK'] = df3['new_page'] * df3['UK']
# ```

# In[38]:


df_new['new_CA'] = df_new['ab_page'] * df_new['CA']
df_new['new_US'] = df_new['ab_page'] * df_new['US']
df_new.head()


# In[39]:


lm2 = sm.Logit(df_new['converted'], df_new[['intercept','ab_page','CA','US','new_CA','new_US']])
results2 = lm2.fit()
results2.summary()


# **P值依然很大，加入交叉项无助于改善模型**

# In[40]:


# Convert timestamp to datetime object
df2['timestamp']=pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
df2.dtypes


# In[41]:


# Duration of experiment
sorted_time = df2['timestamp'].sort_values()
sorted_time[0] - sorted_time[len(sorted_time)-1]  


# **该A/B test实验时长为14天零7小时**

# <a id='conclusions'></a>
# ## 总结
# 
# 
# **1.通过第一部分的单尾假设检验结果，我们没有拒绝原假设(旧页面优于或等于新页面)，建议保留页面的旧版本。**
# **2.逻辑分析结果表明，新旧页面对转换率没有显著性影响（旧页面等于新页面），与第一部分的结论相符**
# **3.加入新因素-国家并不对转换率产生影响，无助于改善模型**
# **4.实验持续时间约为14天，时间的影响可能是决定结论的主要因素，A/B test的分析和结果局限于现有数据以及较短时间的实验——变化厌恶和新奇效应可能会影响结果。建议延长实验时间。**
# 
# 
