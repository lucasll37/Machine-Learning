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
import pickle # salva variaveis em disco
from plotly.offline import plot
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split




"""
BASE DE DADOS "Censo"
"""


# Importação da base de dados

base_census = pd.read_csv("../Bases de dados/census.csv")  #1 <<<<<<<<


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

X_census = base_census.iloc[:, 0:14].values  #2 <<<<<<<<
Y_census = base_census.iloc[:, 14].values  #3 <<<<<<<<



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



# Divisão de bases em treinamento e teste

X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = train_test_split(X_census, Y_census, test_size=0.15, random_state = 0)
#print(X_census_treinamento.shape)
#print(Y_census_treinamento.shape)
#print(X_census_teste.shape)
#print(Y_census_teste.shape)



# Salvando Variáveis

with open('census.pkl', mode='wb') as file:
    pickle.dump([X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste], file)