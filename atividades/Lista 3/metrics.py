import math

def mse(y_true, y_pred):
    cont = 0
    for i in range(len(y_true)):
        cont += (y_true[i] -  y_pred[i]) ** 2
    return 1/len(y_pred) * cont

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true,y_pred))

def mae(y_true, y_pred):
    cont = 0
    for i in range(len(y_true)):
        cont += abs(y_true[i] -  y_pred[i])
    return 1/len(y_pred) * cont