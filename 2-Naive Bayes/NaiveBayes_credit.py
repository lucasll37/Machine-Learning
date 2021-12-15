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


with open('credit.pkl', 'rb') as file:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(file)
    
#print(X_credit_treinamento.shape, X_credit_teste.shape, Y_credit_treinamento.shape, Y_credit_teste.shape)

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, Y_credit_treinamento)
#print(naive_credit_data.predict([[-1.37545, 0.506311, 0.109809]])) # caso index:0
previsoes = naive_credit_data.predict(X_credit_teste)
#print(Y_credit_teste)
#print(accuracy_score(Y_credit_teste, previsoes))
#print(confusion_matrix(Y_credit_teste, previsoes))
#print(classification_report(Y_credit_teste, previsoes))
#print(naive_credit_data.classes_)
#print(naive_credit_data.class_count_)
#print(naive_credit_data.class_prior_)