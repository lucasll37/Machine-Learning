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



with open('census.pkl', 'rb') as file:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(file)
    
#print(X_census_treinamento.shape, X_census_teste.shape, Y_census_treinamento.shape, Y_census_teste.shape) #
random_forest_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 1)
random_forest_census.fit(X_census_treinamento, Y_census_treinamento)
previsoes = random_forest_census.predict(X_census_teste)
#print(previsoes) #
#print(accuracy_score(Y_census_teste, previsoes)) #
#print(confusion_matrix(Y_census_teste, previsoes)) #
print(classification_report(Y_census_teste, previsoes)) #
#print(random_forest_census.feature_importances_) #
#print(random_forest_census.classes_) # 