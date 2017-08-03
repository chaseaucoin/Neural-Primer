import numpy as np

def sigmoid(x):    
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return (x*(1-x))

#input data
X = np.array([[0,0,1],  # Note: there is a typo on this line in the video
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
             [1],
             [1],
             [0]])

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

for j in range(60000):  
    
    # Calculate forward through the network.
    inputLayer = X
    hiddenLayer = sigmoid(np.dot(inputLayer, syn0))
    outputLayer = sigmoid(np.dot(hiddenLayer, syn1))
    
    # Back propagation of errors using the chain rule. 
    outputLayer_error = y - outputLayer
    if(j % 1000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(outputLayer_error))))
        
    outputLayer_delta = outputLayer_error*sigmoidPrime(outputLayer)
    
    hiddenLayer_error = outputLayer_delta.dot(syn1.T)
    
    hiddenLayer_delta = hiddenLayer_error * sigmoidPrime(hiddenLayer)
    
    #update weights (no learning rate term)
    syn1 += hiddenLayer.T.dot(outputLayer_delta)
    syn0 += inputLayer.T.dot(hiddenLayer_delta)
    
print("Output after training")
print(outputLayer)