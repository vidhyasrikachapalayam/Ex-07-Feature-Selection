# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

#CODE

NAME:VIDHYASRI.K

REGISTER NO.: 212222230170

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

p= []

X_1 = X[cols]

X_1 = sm.add_constant(X_1)

model = sm.OLS(y,X_1).fit()

p = pd.Series(model.pvalues.values[1:],index = cols)

pmax = max(p)

feature_with_p_max = p.idxmax()

if(pmax>0.05):

cols.remove(feature_with_p_max)
else:

break
selected_features_BE = cols
print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

model = LinearRegression()

rfe = RFE(model,step=nof_list[n])

X_train_rfe = rfe.fit_transform(X_train,y_train)

X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe,y_train)

score = model.score(X_test_rfe,y_test)

score_list.append(score)

if(score>high_score):

high_score = score

nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()

# OUTPUT
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/76632172-d84e-4c5f-8da8-d8530c6121d6)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/74cbaffb-d3e5-45b9-b2b1-27febe4f1dfe)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/26a26f78-f8b2-44ba-b3e9-5053e2088558)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/f9c38e5b-520d-4077-8a86-f06c2b67d498)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/30d535ee-b525-4c7c-8546-600499dbd268)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/3cc4c6d4-affd-49ab-bdc0-b465e5fa6c8e)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/cbd2106a-ec46-4933-812c-101de1057155)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/1e345eda-4f92-4e0d-8a9d-ddcc59a7f3eb)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/2c9ddb3c-829d-4236-9d84-bba03fae8e7a)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/b70636f9-f942-4d2c-9e51-06885344baa7)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/96b6324a-5a38-4d00-b7c8-08e4b0f881a1)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/bf65b66e-66c5-4dc2-b7de-7f7afa7188ec)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/6fd6a2f4-f7a3-4b0f-b64f-47db5c2292b7)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/8846c408-7b2f-4f40-baf8-2c481f23a1d9)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/d7768d54-6e1d-4134-b42b-e8bceed92431)

![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/7c2ccb22-a58d-4ba9-9649-0995703f3ad7)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/0183054d-96bf-4a6b-9364-0e7da5cedea2)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/09724da2-feab-4b34-8e80-cfbf1d7d5e76)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/4c9e1e45-a4d9-4e16-88f3-8fb2859fb64f)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/c91f5089-cce0-4b37-80d3-8b1c9fa83f6b)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/6c8d914f-674a-4f43-9b6e-0ab53cdf500d)
![image](https://github.com/vidhyasrikachapalayam/Ex-07-Feature-Selection/assets/119477817/4ca49749-4a8a-4ecb-88b2-e6b30411304c)

# RESULT

The various feature selection techniques has been performed on a dataset



