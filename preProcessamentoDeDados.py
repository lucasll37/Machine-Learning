# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 16:17:11 2021

@author: lucas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer



"""
BASE DE DADOS "Crédito"
"""



# Importação da base de dados

#base_credit = pd.read_csv("Bases de dados/credit_data.csv")  #1 <<<<<<<<



# Ferramentas de visualização de base de dados (textual)

#print(base_credit.head())
#print(base_credit.tail())
#print(base_credit.describe())
#print(base_credit[base_credit['income'] >= 69995])
#print(base_credit[base_credit['loan'] <= 1.38])
#print(np.unique(base_credit['default']))
#print(np.unique(base_credit['default'], return_counts=True))
#print(base_credit.mean())
#print(base_credit['loan'].min())
#print(base_credit['loan'].max())
#print(base_credit.columns)



# Ferramentas de visualização de base de dados (gráfico)

#sns.countplot(x=base_credit['default'])
#plt.hist(x=base_credit['age'])
#plt.hist(x=base_credit['loan'])



# Ferramentas de visualização de base de dados (gráfico dinâmico)

#grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default') #dispersão
#plot(grafico, auto_open=True) # ou grafico.show()



# Tratamento de dados

#print(base_credit.loc[base_credit['age'] < 0])
#print(base_credit[base_credit['age'] < 0])
#print(base_credit['age'].mean()) # media de idade errada: 40.8075
#print(base_credit['age'][base_credit['age'] > 0].mean())  # media de idade correta: 40.92
#print(base_credit[base_credit['age'] < 0].index)
#base_credit2 = base_credit.drop('age', axis=1) #apaga fileira:: 0:linha 1:coluna
#base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
#base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92 # preenchimento manual  #2 <<<<<<<<
#print(base_credit.loc[15])
#print(base_credit.loc[21])
#print(base_credit.loc[26])
#print(base_credit.isnull())
#print(base_credit.isnull().sum())
#print(base_credit.loc[pd.isnull(base_credit['age'])])
#base_credit['age'].fillna(base_credit['age'].mean(), inplace=True) # preenchimento automático  #3 <<<<<<<<
#print(base_credit.isnull().sum())
#print(base_credit.loc[base_credit['clientid'].isin([29, 30, 32])])  



# Divisão dos dados entre previsores e classe

#X_credit = base_credit.iloc[0:2000, 1:4].values # ou [:, 1:4]  #4 <<<<<<<<
#Y_credit = base_credit.iloc[0:2000, 4].values  #5 <<<<<<<<



# Escalonamento de valores

#print(X_credit)
#scaler_credit = StandardScaler() #padronização  #6 <<<<<<<<
#X_credit = scaler_credit.fit_transform(X_credit) #padronização  #7 <<<<<<<<
#print(X_credit)



# ------------------------------------------------------------------------

"""
BASE DE DADOS "Censo"
"""


# Importação da base de dados

base_census = pd.read_csv("Bases de dados/census.csv")



# Ferramentas de visualização de base de dados (textual)

#print(base_census.describe())
#print(base_census.head())
#print(base_census.tail())
#print(base_census.mean())
#print(np.unique(base_census['income'], return_counts=True))
#print(base_census['age'].min())
#print(base_census['age'].max())
#print(base_census.columns)



# Ferramentas de visualização de base de dados (gráfico)

#sns.countplot(x=base_census['income'])
#plt.hist(x=base_census['age'])
#plt.hist(x=base_census['education-num'])
#plt.hist(x=base_census['hour-per-week'])



# Ferramentas de visualização de base de dados (gráfico dinâmico)


#grafico1 = px.treemap(base_census, path=["occupation", "relationship", 'age'])
#plot(grafico1, auto_open=True)
#grafico2 = px.parallel_categories(base_census, dimensions=["education", "income"])
#plot(grafico2, auto_open=True)



# Tratamento de dados

#print(base_credit.isnull())
#print(base_credit.isnull().sum())



# Divisão dos dados entre previsores e classe

X_census = base_census.iloc[:, 0:14].values
Y_census = base_census.iloc[:, 14].values



# Tratamento de dados categóricos (LabelEncoder)

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])
#print(X_census[0])



# Tratamento de dados categóricos (OneHotEncoder)

#print(np.unique(base_census['workclass']))
#print(len(np.unique(base_census['workclass'])))
onehotencoder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder="passthrough")
X_census = onehotencoder_census.fit_transform(X_census).toarray()
#print(X_census[0])
#print(X_census.shape)



# Escalonamento de valores

#print(X_census)
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)
#print(X_census)