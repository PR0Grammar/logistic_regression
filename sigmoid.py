import math

def sigmoid(z):
    if(z < 0):
        return 1.0 - ( (1.0) /(1.0 + (math.exp(z))) )
    else:
        return (1.0) /(1.0 + (math.exp(-z)))
