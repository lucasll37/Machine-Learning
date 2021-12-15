# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:52:43 2021

@author: lucas
"""
"""
Random Forest
"""

import matplotlib.pyplot as plt
import pickle # salva variaveis em disco
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier





with open('credit.pkl', 'rb') as file:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(file)
    
#print(X_credit_treinamento.shape, X_credit_teste.shape, Y_credit_treinamento.shape, Y_credit_teste.shape) #
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state = 0)
random_forest_credit.fit(X_credit_treinamento, Y_credit_treinamento)
#(random_forest_credit.predict([[-1.37545, 0.506311, 0.109809]])) #
previsoes = random_forest_credit.predict(X_credit_teste)
#print(previsoes) #
#print(accuracy_score(Y_credit_teste, previsoes)) #
#print(confusion_matrix(Y_credit_teste, previsoes)) #
print(classification_report(Y_credit_teste, previsoes)) #
#print(arvore_credit.feature_importances_) #
#print(random_forest_credit.classes_) #