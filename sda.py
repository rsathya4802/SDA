


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./dataset_Facebook.csv',sep=';')

data.head()

data.iloc[:5,:5]

data.isnull()

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.countplot(x='Category',hue='Type',data=data,)

sns.countplot(x='Category',hue='Post Month',data=data,palette='deep')

data['Lifetime Post Total Reach'].hist(bins=10,color='darkred')

data.info()

data['Category'].value_counts()

data['Post Month'].value_counts()



"""# PCA"""

categorical_data = ['Type','Category','Post Month',	'Post Weekday',	'Post Hour','Paid']

data_pca = data.drop(columns=categorical_data)

plt.figure(figsize=(16,12))
sns.heatmap(data_pca.corr(),annot=True,cbar=True,cmap='YlOrBr')

scaler = StandardScaler()
scaler.fit(data_pca)
data_std = pd.DataFrame( scaler.transform(data_pca),columns=data_pca.columns)

plt.figure(figsize=(16,12))
sns.heatmap(data_std.corr(),annot=True,cbar=True,cmap='YlOrBr')

e_values , e_vectors = np.linalg.eig(data_std.cov().values)

plt.plot(sorted(e_values,reverse=True))

np.sum(e_values[:7])/np.sum(e_values)

"""The 13 features ( Excluding categorical data )  can be approximated to 7 features which contribute to almost 97% of the feature data """



















"""## Pre-processing
Removing rows with missing information
"""

data.dropna(inplace=True)

"""## Regression"""

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest

"""#### Preprocessing for Regression

Removing Outliers
"""

outlier=np.percentile(data['Lifetime Post Total Reach'],90)
outlier

df = data[data['Lifetime Post Total Reach']<outlier]

def Weekday(x):
    if x == 1:
        return 'Sunday'
    elif x== 2:
        return 'Monday'
    elif x == 3:
        return 'Tuesday'
    elif x == 4:
        return 'Wednesday'
    elif x == 5:
        return 'Thursday'
    elif x ==6:
        return 'Friday'
    elif x == 7:
        return "Saturday"

data['Weekday'] = data['Post Weekday'].apply(lambda x: Weekday(x))

df = pd.concat([df,pd.get_dummies(df['Weekday'])],axis=1)

df = pd.concat([df,pd.get_dummies(df['Post Hour'],prefix='hour')],axis=1)
df = pd.concat([df,pd.get_dummies(df['Post Month'],prefix='Month')],axis=1)
df['Video'] = pd.get_dummies(df['Type'])['Video']
df['Status'] = pd.get_dummies(df['Type'])['Status']
df['Photo'] = pd.get_dummies(df['Type'])['Photo']
df['Category_1'] = pd.get_dummies(df['Category'])[1]
df['Category_2'] = pd.get_dummies(df['Category'])[2]

df.info()

"""### Model 1 - Lifetime engaged users"""

x = df[['Page total likes','Paid','Video','Status','Photo','Total Interactions',
    'Category_1','Category_2','Monday','Tuesday','Wednesday',"Thursday",'Friday','Saturday','Sunday',
       'hour_17','hour_1','hour_2','hour_3','hour_4','hour_5', 'hour_6','hour_7','hour_8',
        'hour_9','hour_10','hour_11','hour_12','hour_13','hour_14','hour_15','hour_16',
        'hour_18','hour_19','hour_20','hour_22','hour_23',
        'Month_1','Month_2','Month_3','Month_4','Month_5','Month_6',
        'Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]

y = df['Lifetime Engaged Users']

bestAttributes = SelectKBest(k=49)
bestAttributes.fit(x,y)

temp = x.columns
new = [temp[i] for i in bestAttributes.get_support(indices=True)]

x = df[new]
x_train,x_test,y_train, y_test = train_test_split(x,
                                                  y, test_size=0.1,
                                                  random_state=40)

reg = linear_model.LinearRegression(normalize=True)
reg.fit(x_train,y_train)
lasso = linear_model.Lasso(normalize=True)
lasso.fit(x_train,y_train)

lassoDF = pd.DataFrame()
lassoDF['Attribute'] = list(x_train.columns)
lassoDF['Importance'] = lasso.coef_
lassoDF

"""Regression Model"""

def Regression(model,x_train= None,y_train=None, x_test=None,y_test=None,saveFig=False): 
    
    predicted_test = model.predict(x_test)
    test_score = r2_score(y_test, predicted_test)
    predicted_train = model.predict(x_train)
    train_score = r2_score(y_train, predicted_train)
    print(f'Train data R-2 score: {train_score}')
    print(' ')
    print(f'Test data R-2 score: {test_score}')
    
    DF = pd.DataFrame()
    DF['Score'] = [round(train_score,3),round(test_score,3),]
    DF['Step'] = ['train','test']
    DF['metric'] = ['r2','r2']

    #plotting results
    sns.pointplot(y=DF['Score'],x=DF['Step'],hue=DF['metric'])
    plt.ylim([-.1,1])
    plt.title('Model Scores')
    plt.show()

linear = Regression(reg,x_test=x_test,x_train=x_train,y_test=y_test,y_train=y_train)

pred = reg.predict(x_test)
error = y_test - pred
plt.scatter(pred, error)

import statsmodels.api as sm

x_train = sm.add_constant(x_train)
results = sm.OLS(y_train, x_train).fit()
results.summary()

"""### Model 2 - Features important for Page Likes"""

x = df[['Paid','Video','Status','Photo','Total Interactions','Lifetime Post Total Reach','Lifetime Post Total Impressions',
        'Lifetime Engaged Users',
    'Category_1','Category_2','Monday','Tuesday','Wednesday',"Thursday",'Friday','Saturday','Sunday',
       'hour_17','hour_1','hour_2','hour_3','hour_4','hour_5', 'hour_6','hour_7','hour_8',
        'hour_9','hour_10','hour_11','hour_12','hour_13','hour_14','hour_15','hour_16',
        'hour_18','hour_19','hour_20','hour_22','hour_23',
        'Month_1','Month_2','Month_3','Month_4','Month_5','Month_6',
        'Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]

y = df['Page total likes']

bestAttributes = SelectKBest(k=8)
bestAttributes.fit(x,y)

temp = x.columns
new = [temp[i] for i in bestAttributes.get_support(indices=True)]

x = df[new]
x_train,x_test,y_train, y_test = train_test_split(x,
                                                  y, test_size=0.1,
                                                  random_state=40)

reg = linear_model.LinearRegression(normalize=True)
reg.fit(x_train,y_train)
lasso = linear_model.Lasso(normalize=True)
lasso.fit(x_train,y_train)

lassoDF = pd.DataFrame()
lassoDF['Attribute'] = list(x_train.columns)
lassoDF['Importance'] = lasso.coef_
lassoDF

def Regression(model,x_train= None,y_train=None, x_test=None,y_test=None,saveFig=False): 
    
    predicted_test = model.predict(x_test)
    test_score = r2_score(y_test, predicted_test)
    predicted_train = model.predict(x_train)
    train_score = r2_score(y_train, predicted_train)
    print(f'Train data R-2 score: {train_score}')
    print(' ')
    print(f'Test data R-2 score: {test_score}')
    
    DF = pd.DataFrame()
    DF['Score'] = [round(train_score,3),round(test_score,3),]
    DF['Step'] = ['train','test']
    DF['metric'] = ['r2','r2']

    #plotting results
    sns.pointplot(y=DF['Score'],x=DF['Step'],hue=DF['metric'])
    plt.ylim([-.1,1])
    plt.title('Model Scores')
    plt.show()

linear = Regression(reg,x_test=x_test,x_train=x_train,y_test=y_test,y_train=y_train)

pred = reg.predict(x_test)
error = y_test - pred
plt.scatter(pred, error)

x_train = sm.add_constant(x_train)
results = sm.OLS(y_train, x_train).fit()
results.summary()



"""## EDA"""

plt.figure(figsize=(16,12))
sns.heatmap(data.corr(),annot=True,cbar=True,cmap='YlOrBr')

plt.figure(figsize=(10.5,6))
sns.distplot(data['Page total likes'],bins=20,kde=True,color="black")
# plt.xlim([6000,14000])
plt.title("Page total likes",fontsize=15)



plt.figure(figsize=(10.5,6))
sns.distplot(data['like'],bins=100,color='black',kde=True)
plt.xlim(0,800)
plt.xlabel("NUMBER OF LIKES",fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.title('Like - Post',fontsize=15)

plt.figure(figsize=(10.5,6))
sns.distplot(data['Lifetime Engaged Users'],bins=100,color='black')
plt.xlim(0,4000)
plt.title('Lifetime engaged users',fontsize=15)

"""The distribution is left skewed with most posts with engagement around 500 users, with the maximum being around 12000."""

plt.figure(figsize=(10.5,6))
sns.distplot(data['Lifetime Post Total Reach'],bins=200,color='black')
plt.xlim(0,100000)

"""Similarly, the distribution is left skewed, with majority of posts reaching 0-10000 users."""

fig, ax = plt.subplots(ncols=3,nrows=1,sharey=True,figsize=(24,9))

paid = data[data['Paid']==1]
free = data[data['Paid']==0]

ax[0].scatter(free['like'],free['Lifetime Engaged Users'],color='y')
ax[0].scatter(paid['like'],paid['Lifetime Engaged Users'],color='b')
ax[0].set_title('Likes')
ax[0].set_xlim(0,1250)
ax[0].legend(labels=['Free','Paid'])

ax[1].scatter(free['comment'],free['Lifetime Engaged Users'],color='y')
ax[1].scatter(paid['comment'],paid['Lifetime Engaged Users'],color='b')
ax[1].set_title('Comments')
ax[1].set_xlim(0,100)
ax[1].legend(labels=['Free','Paid'])

ax[2].scatter(free['share'],free['Lifetime Engaged Users'],color='y')
ax[2].scatter(paid['share'],paid['Lifetime Engaged Users'],color='b')
ax[2].set_title('Shares')
ax[2].set_xlim(0,150)
ax[2].legend(labels=['Free','Paid'])

ax[0].set_ylabel("Lifetime reach")

fig.suptitle('Engagement Metrics vs. Lifetime Engaged Users',fontsize=15)

"""There is a visible relationship between Like/Share and Total users reached, but the number of comments is little more random."""

plt.figure(figsize=(6,6))
sns.boxplot(x=data['Category'],y=data['Lifetime Engaged Users'],palette='rocket')
plt.ylim(0,3000)

plt.figure(figsize=(6,6))
sns.boxplot(x=data['Type'],y=data['Lifetime Engaged Users'],palette='icefire')
plt.ylim(0,5000)

plt.figure(figsize=(10,6))
sns.boxplot(x=data['Type'],y=data['Lifetime Engaged Users'],hue=data['Paid'],palette='rocket')
plt.ylim(0,5000)

"""Engagement for Status > Video > photo > Link"""

sns.lmplot(x='Total Interactions',y='Lifetime Engaged Users',
           hue='Type',data=data,scatter_kws= {'alpha':0.5},palette='icefire')
plt.title('Total Interactions vs. Lifetime Engaged Users')
plt.xlim(-100,2000)
plt.ylim(-300,6500)

"""Interactions : Engaged Users ratio is higher for Photo than others"""

plt.style.use('ggplot')
sns.lmplot(x='Total Interactions',y='Lifetime Engaged Users',
           hue='Paid',data=data,fit_reg=False,scatter_kws= {'alpha':0.7})
plt.title('Total Interactions vs. Lifetime Engaged Users')
plt.xlim(-100,2500)
plt.ylim(-300,6000)

plt.style.use('ggplot')
sns.lmplot(x='Page total likes',y='Lifetime Engaged Users',
           hue='Type',data=data,fit_reg=False,scatter_kws= {'alpha':0.7},)
plt.title('Page Likes vs. Lifetime Engaged Users')
# plt.xlim(-100,140000)
plt.ylim(-300,6500)

sns.boxplot(x=data['Paid'],y=data['Lifetime Engaged Users'],palette='rocket')
plt.ylim(0,2300)

"""Paid posts have slightly higher engagement than non-paid posts."""

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Post Weekday'],y=data['Lifetime Engaged Users'],palette='Spectral')
plt.ylim(0,2000)
plt.title("Post Engagement by Weekday")

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Post Weekday'],y=data['Lifetime Engaged Users'],hue=data['Paid'],palette='rocket')
plt.ylim(0,2000)
plt.title("Post Engagement by Weekday")

"""Paid posts get higher engagement than non-paid posts."""

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Post Weekday'],y=data['like'],palette='Spectral')
plt.ylim(-25,500)
plt.title("Post likes by Weekday")

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Post Weekday'],y=data['like'],hue=data['Paid'],palette='rocket')
plt.ylim(-25,650)
plt.title("Post likes by Weekday")

"""Paid posts get higher likes than non-paid posts."""

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Type'],y=data['like'],hue=data['Paid'],palette='rocket')
plt.ylim(-25,500)
plt.title("Post likes by Weekday")

"""Paid posts get higher likes in general."""

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Type'],y=data['like'],palette='icefire')
plt.ylim(0,500)

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Type'],y=data['comment'],hue=data['Paid'],palette='icefire')
plt.ylim(0,100)

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Type'],y=data['like'],hue=data['Paid'],palette='icefire')
plt.ylim(0,500)

a = data['Lifetime Post reach by people who like your Page']/data['Lifetime Post Total Reach']
new = []
for i in range(len(a)):
  new.append(np.mean(a[:i]))
plt.ylabel('Fraction of reach by people who liked the page')
plt.plot(new)
print(f"{np.mean(a)}")

"""65 % of total posts reach are people that have already liked the page."""

a = data['Lifetime People who have liked your Page and engaged with your post']/data['Lifetime Engaged Users']

new = []
for i in range(len(a)):
  new.append(np.mean(a[:i]))

plt.ylabel('Fraction of engaged users who liked the page')

plt.plot(new)
print(f"{round(np.mean(a),2)}")

"""70% of total engagement in posts are from people that have already liked the page."""

a = data['Lifetime Post Impressions by people who have liked your Page']/data['Lifetime Post Total Impressions']
new = []
for i in range(len(a)):
  new.append(np.mean(a[:i]))
plt.ylabel('Fraction of impression by people who liked the page')
plt.plot(new)
print(f"{np.mean(a)}")

sns.lmplot(x='Page total likes',y='like',
           hue='Paid',data=data,scatter_kws= {'alpha':0.4},palette='dark')
plt.title('Page Likes - Post Likes')
plt.ylim(0,600)

"""No increase in likes for non-paid posts, weak positive trend for paid posts."""

data



"""Relations between like comment and share"""

sns.regplot(x="like",y="comment",data=data);
plt.ylim(0,150)
plt.xlim(0,2100)

sns.regplot(x="like",y="share",data=data);
plt.xlim(0,2100)
plt.ylim(0,220)

sns.regplot(x="share",y="comment",data=data);
plt.ylim(0,150)
plt.xlim(0,220)



"""How much fraction of 'Reach', 'Impression' and 'Engagement' comes from people who liked the page?"""

a = data['Lifetime Post reach by people who like your Page']/data['Lifetime Post Total Reach']
sns.regplot(x="Page total likes",y=a, data=data);
plt.ylabel('Fraction of reach by people who liked the page')

a = data['Lifetime People who have liked your Page and engaged with your post']/data['Lifetime Engaged Users']
sns.regplot(x="Page total likes",y=a, data=data);
plt.ylabel('Fraction of engaged users who liked the page')

a = data['Lifetime Post Impressions by people who have liked your Page']/data['Lifetime Post Total Impressions']
sns.regplot(x="Page total likes",y=a, data=data);
plt.ylabel('Fraction of impression by people who liked the page')

"""Which attributes are least affected and most affected by 'Paid'?"""

sns.boxplot(x=data['Paid'],y=data['like'],showmeans=True, palette='rocket')
plt.ylim(0,500)

sns.boxplot(x=data['Paid'],y=data['comment'],showmeans=True,palette='rocket')
plt.ylim(0,25)

sns.boxplot(x=data['Paid'],y=data['share'],showmeans=True,palette='rocket')
plt.ylim(0,100)

"""
> Indented block

"""

sns.boxplot(x=data['Paid'],y=data['Lifetime Post Total Reach'],showmeans=True,palette='rocket')
plt.ylim(0,60000)

sns.boxplot(x=data['Paid'],y=data['Lifetime Post Consumers'],showmeans=True,palette='rocket')
plt.ylim(0,2500)





sns.boxplot(x=data['Paid'],y=data['Lifetime Post Total Impressions'],showmeans=True,palette='rocket')
plt.ylim(0,100000)

# QQ Plot

qqplot(data['like'], line='s')
plt.ylabel('like')
plt.show()





qqplot(data['Lifetime Post Total Reach'], line='s')
plt.ylabel('Lifetime Post Total Reach')
plt.show()

qqplot(data['Lifetime Engaged Users'], line='s')
plt.ylabel('Lifetime Engaged Users')
plt.show()

qqplot(data['Page total likes'], line='s')
plt.ylabel('Page total likes')
plt.show()

qqplot(data['share'], line='s')
plt.ylabel('share')
plt.show()

qqplot(data['comment'], line='s')
plt.ylabel('comment')
plt.show()



data1 = data.sort_values('Page total likes')
data1 = data1.reset_index(drop=True)
plt.ylabel('Total page likes')
plt.xlabel('Sorted')
plt.plot(data1['Page total likes'])

data1 = data.sort_values('like')
data1 = data1.reset_index(drop=True)
a = data1['like']
plt.ylabel('likes')
plt.xlabel('Sorted')
plt.plot(a)



data1 = data.sort_values('Lifetime Post Total Impressions')
data1 = data1.reset_index(drop=True)
a = data1['Lifetime Post Total Impressions']
plt.ylabel('Lifetime Post Total Impressions')
plt.xlabel('Sorted')
plt.plot(a)

data1 = data.sort_values('Lifetime Engaged Users')
data1 = data1.reset_index(drop=True)
a = data1['Lifetime Engaged Users']
plt.ylabel('Lifetime Engaged Users')
plt.xlabel('Sorted')
plt.plot(a)

data.head()

ax = sns.barplot(x="Post Hour", y="Lifetime Post Consumers", data=data)

"""Influence of hour on lifetime consumers

"""

plt.bar("Category","Lifetime Post Consumers", color="Red",data=data)
plt.xlabel("Category")
plt.ylabel("Life Time Post Consumers")

"""Influence of category on life time post consumers"""

ax = sns.barplot(x="Post Weekday", y="Lifetime Post Consumers", data=data)

"""Influence of weekday on life time post consumers"""

ax = sns.barplot(x="Type", y="Lifetime Post Consumers", data=data)

"""Influence of Type on life time post consumers"""

ax = sns.barplot(x="Post Month", y="Lifetime Post Consumers", data=data)

"""Influence of month on lifetime post consumers"""

ax = sns.barplot(x="Paid", y="Lifetime Post Consumers", data=data)

"""Influence of paid on lifetime post time consumers"""

plt.figure(figsize=(10.5,6))
sns.distplot(data['share'],bins=100,color='black',kde=True)
plt.xlim(0,800)
plt.xlabel("NUMBER OF Shares",fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.title('Share - Post',fontsize=15)

plt.figure(figsize=(10.5,6))
sns.distplot(data['Lifetime Post Consumers'],bins=100,color='black')
plt.xlim(0,4000)
plt.title('Lifetime Post Consumers',fontsize=15)

"""The distribution is left skewed with most posts  around 500 users, with the maximum being around 12000"""

fig, ax = plt.subplots(ncols=3,nrows=1,sharey=True,figsize=(24,9))

paid = data[data['Paid']==1]
free = data[data['Paid']==0]

ax[0].scatter(free['like'],free['Lifetime Post Consumers'],color='y')
ax[0].scatter(paid['like'],paid['Lifetime Post Consumers'],color='b')
ax[0].set_title('Likes')
ax[0].set_xlim(0,1250)
ax[0].legend(labels=['Free','Paid'])

ax[1].scatter(free['comment'],free['Lifetime Post Consumers'],color='y')
ax[1].scatter(paid['comment'],paid['Lifetime Post Consumers'],color='b')
ax[1].set_title('Comments')
ax[1].set_xlim(0,100)
ax[1].legend(labels=['Free','Paid'])

ax[2].scatter(free['share'],free['Lifetime Post Consumers'],color='y')
ax[2].scatter(paid['share'],paid['Lifetime Post Consumers'],color='b')
ax[2].set_title('Shares')
ax[2].set_xlim(0,150)
ax[2].legend(labels=['Free','Paid'])

ax[0].set_ylabel("Lifetime reach")

fig.suptitle('Engagement Metrics vs. Lifetime Post Consumers',fontsize=15)

plt.figure(figsize=(10,6))
sns.boxplot(x=data['Type'],y=data['Lifetime Post Consumers'],hue=data['Paid'],palette='rocket')
plt.ylim(0,5000)

"""Engagement for Status > Video > photo > Link"""

sns.lmplot(x='Total Interactions',y='Lifetime Post Consumers',
           hue='Type',data=data,scatter_kws= {'alpha':0.5},palette='icefire')
plt.title('Total Interactions vs. Lifetime Post Consumers')
plt.xlim(-100,2000)
plt.ylim(-300,6500)

"""Interactions : consumers ratio is higher for Photo than others"""

plt.style.use('ggplot')
sns.lmplot(x='Total Interactions',y='Lifetime Post Consumers',
           hue='Paid',data=data,fit_reg=False,scatter_kws= {'alpha':0.7})
plt.title('Total Interactions vs. Lifetime Post Consumers')
plt.xlim(-100,2500)
plt.ylim(-300,6000)

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Type'],y=data['share'],hue=data['Paid'],palette='icefire')
plt.ylim(0,500)

plt.figure(figsize=(8,6))
sns.boxplot(x=data['Post Weekday'],y=data['share'],hue=data['Paid'],palette='rocket')
plt.ylim(-25,650)
plt.title("Shares in Weekday")

data.groupby(['Type']).sum().plot(kind='pie', y='Page total likes', autopct='%1.1f%%')

data.groupby(['Type']).sum().plot(kind='pie', y='Lifetime Engaged Users', autopct='%1.1f%%')

data.groupby(['Paid']).sum().plot(kind='pie', y='Page total likes', autopct='%1.1f%%')

data.groupby(['Type']).sum().plot(kind='pie', y='like', autopct='%1.1f%%', figsize=(10, 5))
data.groupby(['Type']).sum().plot(kind='pie', y='comment', autopct='%1.1f%%', figsize=(10, 5))
data.groupby(['Type']).sum().plot(kind='pie', y='share', autopct='%1.1f%%', figsize=(10, 5))



for name in ['Lifetime Post Total Impressions','Total Interactions']:
    data_pca[name].plot(kind='hist',title=name,bins=20)
    plt.show()

data_count = data.groupby("Post Weekday")["Lifetime Post Total Impressions"].sum().sort_values()
data_count.plot(kind="barh",title='Total Impressions based on Weekdays')



