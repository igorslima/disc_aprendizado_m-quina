import numpy as np

def matriz_confusao(y_true, y_pred):
    matriz = [[0,0],[0,0]]
    for x in range(y_true.shape[0]):
        matriz[y_true[x]][y_pred[x]] += 1 
    return matriz

def accuracy(y_true, y_pred):
    mat = matriz_confusao(y_true, y_pred)
    return (mat[0][0] + mat[1][1])/y_true.shape[0]

def recall(y_true, y_pred):
    mat = matriz_confusao(y_true, y_pred)
    return mat[1][1]/(mat[1][1] + mat[1][0])

def precision(y_true, y_pred):
    mat = matriz_confusao(y_true, y_pred)
    return mat[1][1]/(mat[1][1] + mat[0][1])

def f1_measure(y_true, y_pred):
    return 2 * ((precision(y_true, y_pred) * recall(y_true, y_pred)) / (precision(y_true, y_pred) + recall(y_true, y_pred)))