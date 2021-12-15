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
from sklearn.model_selection import train_test_split
import pickle # salva variaveis em disco



"""
BASE DE DADOS "Crédito"
"""



# Importação da base de dados

base_credit = pd.read_csv("../Bases de dados/credit_data.csv")



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
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92 # preenchimento manual
#print(base_credit.loc[15])
#print(base_credit.loc[21])
#print(base_credit.loc[26])
#print(base_credit.isnull())
#print(base_credit.isnull().sum())
#print(base_credit.loc[pd.isnull(base_credit['age'])])
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True) # preenchimento automático
#print(base_credit.isnull().sum())
#print(base_credit.loc[base_credit['clientid'].isin([29, 30, 32])])  



# Divisão dos dados entre previsores e classe

X_credit = base_credit.iloc[0:2000, 1:4].values # ou [:, 1:4]
Y_credit = base_credit.iloc[0:2000, 4].values



# Escalonamento de valores

#print(X_credit)
scaler_credit = StandardScaler() #padronização
X_credit = scaler_credit.fit_transform(X_credit) #padronização
#print(X_credit)



# Divisão de bases em treinamento e teste

X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit, Y_credit, test_size=0.25, random_state = 0)
#print(X_credit_treinamento.shape)
#print(Y_credit_treinamento.shape)
#print(X_credit_teste.shape)
#print(Y_credit_teste.shape)



# Salvando Variáveis

with open('credit.pkl', mode='wb') as file:
    pickle.dump([X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste], file)
    
