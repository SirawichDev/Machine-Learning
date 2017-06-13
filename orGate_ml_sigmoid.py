import numpy as np 

def sigmoid(x, deriv=False): 
	if(deriv == True):
		return (1-x)*x
	return 1/( 1+np.exp( -1*x ) )

np.random.seed(1)

X = np.array (
	[ [ 0,0 ], 
	  [ 0,1 ],
	  [ 1,0 ],
	  [ 1,1 ] ])

Y = np.array ( [ [ 0,1 ,1 ,1 ] ]).T

syn0 = 2*np.random.random( ( 2,1 ) ) - 1

for iter in xrange(10000):

	layer0 = X
	layer1 = sigmoid(np.dot(layer0, syn0) )
	layer1_error = Y - layer1
	layer1_delta = layer1_error * sigmoid( layer1,True )

	syn0 += np.dot( layer0.T , layer1_delta )

print "Output After Training By use Sigmoid Derivative Function: "
print layer1