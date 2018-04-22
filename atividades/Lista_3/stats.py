import numpy as np
import math

def mean(x):
    mean = 0
    for i in x:
        mean += i
    return mean/len(x)
    
def var(x):
    media = mean(x)
    cont = 0
    for elem in x:
        cont += (elem - media) ** 2  
    return cont * 1/len(x)

def stdev(x): 
    return  math.sqrt(var(x))
