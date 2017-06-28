import numpy as np

def sigmoid(x, deriv=False):
    if (deriv==True):
        return (1-x)*x #diff_function sigmoid
    return 1/(1+np.exp(-3*x)) #function sigmoid  diff


np.random.seed(2)
epoch =20000
hidden = 10
alpha =0.9

eta= 0.8
X = np.array (
    [ [0,0],
      [0,1],
      [1,0],
      [1,1] ] )

Y = np.array ( [[ 0,0,0,1] ] ).T

syn0 = 2 * np.random.random((2, hidden)) - 1 #ran 1-2
syn1 =  2 * np.random.random((hidden, 1)) - 1
syn0_prev = 0 * syn0
syn1_prev = 0 * syn1

for iter in xrange(epoch):

    layer0=X #layer0
    layer1 = sigmoid(np.dot( layer0, syn0) ) #output
    layer2 = sigmoid(np.dot( layer1, syn1) )


    layer2_error = Y - layer2
    layer2_delta = eta * layer2_error * sigmoid(layer2,True)

    layer1_error = layer2_delta.dot(syn1.T)
    layer1_delta = layer1_error *  sigmoid(layer1,True)

    syn0_delta = (1 - alpha) * np.dot( layer0.T , layer1_delta ) + alpha * syn0_prev
    syn1_delta =  (1 - alpha) * np.dot( layer1.T , layer2_delta ) + alpha * syn1_prev
    syn1 += syn1_delta
    syn0 += syn0_delta

    syn1_prev = syn1_delta
    syn0_prev = syn0_delta


print "Output After Training: "
print layer2
