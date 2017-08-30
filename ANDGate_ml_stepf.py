import numpy as np



def step(x, deriv=False):
	if (deriv == True):
		return 1
	return np.sign(x)

np.random.seed(1)

X = np.array (
	[ [ 0,0 ], 
	  [ 0,1 ],
	  [ 1,0 ],
	  [ 1,1 ] ])

Y = np.array ( [ [ 0,1 ,1 ,1 ] ]).T

syn0 = 2*np.random.random( ( 2,1 ) ) - 1

for iter in range(10000):

	layer0 = X
	layer1 = step(np.dot(layer0, syn0) )
	layer1_error = Y - layer1
	layer1_delta = layer1_error * step( layer1,True )

	syn0 += np.dot( layer0.T , layer1_delta )

print ("Output After Training By use Step Derivative Function:")
print (layer1)