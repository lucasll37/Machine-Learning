# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:18:49 2021

@author: lucas
"""

"""
Vantagem:
    > Rápido
    > Simplicidade de interpretação
    > Trabalha com altas dimensôes
    > Boas previsões em bases de dados pequenas
    > Bom desempenho em tarefas de classificação de texto
    
Desvantagem:
    > Combinação de caracteristicas (atributos independentes > cada par de características são independentes (nem sempre é verdade))
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

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


with open('census.pkl', 'rb') as file:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(file)
    
#print(X_census_treinamento.shape, X_census_teste.shape, Y_census_treinamento.shape, Y_census_teste.shape)
naive_census_data = GaussianNB()
naive_census_data.fit(X_census_treinamento, Y_census_treinamento)
previsoes = naive_census_data.predict(X_census_teste)
#print(Y_census_teste)
#print(accuracy_score(Y_census_teste, previsoes))
#print(confusion_matrix(Y_census_teste, previsoes))
print(classification_report(Y_census_teste, previsoes))
#print(naive_census_data.classes_)
#print(naive_census_data.class_count_)
#print(naive_sensus_data.class_prior_)