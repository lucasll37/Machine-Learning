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


with open('credit.pkl', 'rb') as file:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(file)
    
#print(X_credit_treinamento.shape, X_credit_teste.shape, Y_credit_treinamento.shape, Y_credit_teste.shape) #
#print(X_credit_teste.shape)
#print(Y_credit_teste.shape)
#print(X_credit_treinamento.shape)
#print(Y_credit_treinamento.shape)
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(X_credit_treinamento, Y_credit_treinamento)
previsoes = knn_credit.predict(X_credit_teste)
#print(previsoes)
#print(Y_credit_teste)
#print(accuracy_score(Y_credit_teste, previsoes)) #
#print(confusion_matrix(Y_credit_teste, previsoes)) #
print(classification_report(Y_credit_teste, previsoes)) #
