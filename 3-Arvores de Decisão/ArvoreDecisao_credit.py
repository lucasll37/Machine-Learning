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



with open('credit.pkl', 'rb') as file:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(file)
    
#print(X_credit_treinamento.shape, X_credit_teste.shape, Y_credit_treinamento.shape, Y_credit_teste.shape) #
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state = 0)
arvore_credit.fit(X_credit_treinamento, Y_credit_treinamento)
#print(arvore_credit.predict([[-1.37545, 0.506311, 0.109809]])) #
previsoes = arvore_credit.predict(X_credit_teste)
#print(previsoes) #
#print(accuracy_score(Y_credit_teste, previsoes)) #
#print(confusion_matrix(Y_credit_teste, previsoes)) #
#print(classification_report(Y_credit_teste, previsoes)) #
#print(arvore_credit.feature_importances_) #
#print(arvore_credit.classes_) #
previsores = ['income', 'age', 'loan']
fig, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
tree.plot_tree(arvore_credit, feature_names = previsores, class_names = ['0', '1'], filled = True)
#fig.savefig('arvore_credit.png')