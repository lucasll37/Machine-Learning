# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:18:49 2021

@author: lucas

Vantagem:
    > Algoritmo simples e poderoso
    > Indicado quando o relacionamento entre as caracteristicas é complexo
    
Desvantagem:
    > Sensíveis a ruídos e outliers para k pequeno
    > Para k grande: overfitting (default: 3 ou 5)
    > Lento para fazer previsões
    
"""


import matplotlib.pyplot as plt
import pickle # salva variaveis em disco
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


with open('census.pkl', 'rb') as file:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(file)
    
#print(X_census_treinamento.shape, X_census_teste.shape, Y_census_treinamento.shape, Y_census_teste.shape) #
#print(X_census_teste.shape) #
#print(Y_census_teste.shape) #
#print(X_census_treinamento.shape) #
#print(Y_census_treinamento.shape) #
knn_census = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_census.fit(X_census_treinamento, Y_census_treinamento)
previsoes = knn_census.predict(X_census_teste)
#print(previsoes)
#print(Y_census_teste)
print(accuracy_score(Y_census_teste, previsoes)) #
print(confusion_matrix(Y_census_teste, previsoes)) #
print(classification_report(Y_census_teste, previsoes)) #