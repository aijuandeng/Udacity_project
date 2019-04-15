import pickle
import pandas as pd
import  numpy as np

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
df = pd.DataFrame(data_dict)
df = df.T
total_features = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
       'email_address', 'exercised_stock_options', 'expenses', 'from_messages',
       'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances',
       'long_term_incentive', 'other', 'restricted_stock',
       'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
       'to_messages', 'total_payments', 'total_stock_value']
df = df.ix[:,total_features]    # 调整poi为第一列

print('数据集有',len(df.columns)-1,'个特征和',len(df.index),'条员工信息')    # 有21个特征和146条员工信息
poi_count = df['poi'].sum()
print("嫌疑人有:",poi_count)  # 嫌疑人有18个
poi_ratio = round(df['poi'].mean(),2)
print("嫌疑人占人数比：",poi_ratio)
df.replace("NaN",np.nan,inplace=True)   # 将'NaN'替换成nan,使pandas的isnull可以识别到
person_feature_nan = df.isnull().sum(axis=1).sort_values(ascending=False)
feature_nan = df.isnull().sum(axis=0).sort_values(ascending=False)

person_feature_nan = person_feature_nan[person_feature_nan.values > 10].index   # 删除员工信息中特征值缺失超过10个的姓名
feature_nan = feature_nan[:5].index  # 删除特征值缺失最多的前5个特征

df.drop(feature_nan,inplace=True,axis=1)
df.drop(person_feature_nan,inplace=True)

df.fillna(0,inplace=True)
df.drop(["other",'email_address'],inplace=True,axis=1)
df['ratio_from_poi'] = round(df['from_this_person_to_poi']/(df['from_messages']) * 100,2)
df['ratio_to_poi'] = round(df['from_poi_to_this_person']/df['to_messages']* 100, 2)
df.drop(['from_messages','to_messages','from_poi_to_this_person','from_this_person_to_poi'],inplace=True,axis=1)    # 删除多余特征
# print(df.head())

### 散点图
# import matplotlib.pyplot as plt
#
# plt.scatter(df['bonus'],df['salary'])
# plt.show()

outlier = df.iloc[df['salary'].values.argmax()].name
df.drop(outlier,inplace=True)   # 删除异常点
print('处理后的数据集大小为',df.shape)


### 将字典转换为numpy特性数组
from feature_format import featureFormat, targetFeatureSplit

features_list = list(df.columns)
keep_names_list = list(df.index)
dict = {}
for names,data in data_dict.items():
    if data['from_messages'] != "NaN" and data['from_this_person_to_poi'] != "NaN":
        data_dict[names]['ratio_from_poi'] = \
            round(float(data['from_this_person_to_poi'])/(float(data['from_messages'])) * 100,2)
    else:
        data_dict[names]['ratio_from_poi'] = "NaN"
    if data['to_messages'] != "NaN" and data['from_poi_to_this_person'] != "NaN":
        data_dict[names]['ratio_to_poi'] = \
            round(float(data['from_poi_to_this_person'])/(float(data['to_messages'])) * 100, 2)
    else:
        data_dict[names]['ratio_to_poi'] = "NaN"
    if names in keep_names_list:
        dict[names] = data

data = featureFormat(dict, features_list, sort_keys = True)

### 区分标签和特征
from tester import dump_classifier_and_data
labels, features = targetFeatureSplit(data)

### 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### 分类器
from sklearn.metrics import classification_report
def classifiers(clf):
    clf.fit(features_train, labels_train)
    pre = clf.predict(features_test)
    score = clf.score(features_test, labels_test)
    report = classification_report(pre, labels_test)
    print("Accuracy: %s" % score)
    print("Classification report:")
    print(report)
    return clf

from sklearn.tree import DecisionTreeClassifier
print('决策树模型评估：')
clfDT = DecisionTreeClassifier()
classifiers(clfDT)
from sklearn.svm import SVC
print('SVC模型评估：')
clfSVC = SVC()
classifiers(clfSVC)
from sklearn.naive_bayes import GaussianNB
print('贝叶斯模型评估：')
clfNB = GaussianNB()
classifiers(clfNB)
# SVC模型最佳0.81

### 管道
from sklearn.pipeline import Pipeline

def pipe(steps):
    pi = Pipeline(steps).fit(features_train, labels_train)
    pre = pi.predict(features_test)
    score = pi.score(features_test,labels_test)
    report = classification_report(pre, labels_test)
    print(sorted(pi.named_steps['pca'].explained_variance_ratio_)[::-1])  # 有什么用？
    print("Accuracy: %s" % score)
    print("Classification report:")
    print(report)
    return pi

### 特征缩放
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()

### 特征选择
from tester import dump_classifier_and_data
# 1.根据chi2指标选择分数前k个的特征
from sklearn.feature_selection import SelectKBest, chi2
select = SelectKBest(chi2,k=10)

### 降维
from sklearn.decomposition import PCA
pca = PCA(n_components=4)

steps1 = [('scale',scale),('select',select),('pca',pca),('clf',clfDT)]
steps2 = [('select',select),('pca',pca),('clf',clfSVC)]
steps3 = [('scale',scale),('select',select),('pca',pca),('clf',clfNB)]
print('各步骤流水化后： ')
print('决策树模型评估：')
pipeDT = pipe(steps1)

print('SVC模型评估：')
pipeSVC = pipe(steps2)
print('贝叶斯模型评估：')
pipeNB = pipe(steps3)

### 优化参数
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=5)

# cross_validation = kfold.split(labels, features)
cross_validate = cross_val_score(pipeNB,features_train,labels_train,cv=kfold)
print(cross_validate.mean())


paramSVC = {'C':[1,5,10]}   # 只可以选择一组参数？
paramDT = {'min_samples_split':[2,10,15]}

# print(clfSVC)
en = pipeSVC.named_steps['clf']
# en = pipeDT.named_steps['clf']

# grid = GridSearchCV(en,param_grid=paramDT,cv=3)
grid = GridSearchCV(en,param_grid=paramSVC,cv=3)
# print(features_train, labels_train)
grid.fit(features_train, labels_train)

score = grid.score(features_test,labels_test)

pre = grid.predict(features_test)

report = classification_report(pre,labels_test)
print('参数调优')
print('最佳评估器： ', grid.best_estimator_)
print('最佳参数： ', grid.best_params_)
print('最佳分数： ', grid.best_score_)
print("模型评估分数为：" % score)
print("评估报告：")
print(report)



# 分类器中SVM的分类效果最佳
clf = SVC(C=1)
clf.fit(features_train, labels_train)
pre = clf.predict(features_test)
score = clf.score(features_test, labels_test)
report = classification_report(pre, labels_test)
print("Accuracy: %s" % score)
print("Classification report:")
print(report)

steps = [('select',select),('pca',pca),('clf',clf)]
pi = Pipeline(steps).fit(features_train, labels_train)
pre = pi.predict(features_test)
score = pi.score(features_test, labels_test)
report = classification_report(pre, labels_test)
print("Accuracy: %s" % score)
print("Classification report:")
print(report)

cross_validate = cross_val_score(pi,features,labels,cv=kfold)
print(cross_validate.mean())
