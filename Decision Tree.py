# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:38:11 2021

@author: Qalbe
"""

import pandas as pd

data = pd.read_csv("salaries.csv")
data.head()



X= data.drop('salary_more_then_100k', axis = 'columns')
y = data['salary_more_then_100k']

X.head()
y.head()


# converting x data to labels

from sklearn.preprocessing import LabelEncoder
encoder_company = LabelEncoder()
encoder_job = LabelEncoder()
encoder_degree = LabelEncoder()


#converting each column of x to labels
X['company_n'] = encoder_company.fit_transform(X['company'])
X['job_n'] = encoder_job.fit_transform(X['job'])
X['degree_n'] = encoder_degree.fit_transform(X['degree'])

X


X_new = X.drop(['company','job','degree'],axis='columns')

X_new




from sklearn import tree
model = tree.DecisionTreeClassifier()



model.fit(X_new, y)
model.score(X_new, y)

model.predict([[2,2,1]])