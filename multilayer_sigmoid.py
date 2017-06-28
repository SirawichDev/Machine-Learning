import numpy as np

def sigmoid(x, deriv=False):
    if (deriv==True):
        return (1-x)*x #diff_function sigmoid
    return 1/(1+np.exp(-3*x)) #function sigmoid  diff


np.random.seed(2)
epoch =10000
hidden0 = 9
hidden1 = 5

eta= 0.8
X = np.array (
    [ [0,0],
      [0,1],
      [1,0],
      [1,1] ] )

Y = np.array ( [[ 0,0,0,1] ] ).T

syn0 = 2 * np.random.random((2, hidden0)) - 1 #ran 1-2
syn1 = 2 * np.random.random((hidden0, hidden1)) - 1
syn2 =  2 * np.random.random((hidden1, 1)) - 1
for iter in xrange(epoch):

    layer0=X #layer0
    layer1 = sigmoid(np.dot( layer0, syn0) ) #output
    layer2 = sigmoid(np.dot( layer1, syn1) )
    layer3 = sigmoid(np.dot( layer2 , syn2))

    layer3_error = Y - layer3
    layer3_delta = eta * layer3_error * sigmoid(layer3,True)

    layer2_error = layer3_delta.dot(syn2.T)
    layer2_delta = layer2_error * sigmoid(layer2,True)

    layer1_error = layer2_delta.dot(syn1.T)
    layer1_delta = layer1_error *  sigmoid(layer1,True)

    syn0 += np.dot( layer0.T , layer1_delta )
    syn1 += np.dot( layer1.T , layer2_delta )
    syn2 += np.dot( layer2.T , layer3_delta )

print "Output After Training: "
print layer3
