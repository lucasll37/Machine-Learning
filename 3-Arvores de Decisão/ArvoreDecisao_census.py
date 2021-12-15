# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 13:18:49 2021

@author: lucas

Vantagem:
    > Rápido
    > Simplicidade de interpretação
    > Trabalha com altas dimensôes
    > Boas previsões em bases de dados pequenas
    > Bom desempenho em tarefas de classificação de texto
    
Desvantagem:
    > Combinação de caracteristicas (atributos independentes > cada par de características são independentes (nem sempre é verdade))
"""


import matplotlib.pyplot as plt
import pickle # salva variaveis em disco
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



with open('census.pkl', 'rb') as file:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(file)
    
#print(X_census_treinamento.shape, X_census_teste.shape, Y_census_treinamento.shape, Y_census_teste.shape) #
arvore_census = DecisionTreeClassifier(criterion='entropy', random_state = 0)
arvore_census.fit(X_census_treinamento, Y_census_treinamento)
previsoes = arvore_census.predict(X_census_teste)
#print(previsoes) #
#print(accuracy_score(Y_census_teste, previsoes)) #
#print(confusion_matrix(Y_census_teste, previsoes)) #
#print(classification_report(Y_census_teste, previsoes)) #
#print(arvore_census.feature_importances_) #
#print(arvore_census.classes_) #

fig, eixos = plt.subplots(nrows=1, ncols=1, figsize=(50, 50))
tree.plot_tree(arvore_census, class_names = ['0', '1'], filled = True)
fig.savefig('arvore_census.png')