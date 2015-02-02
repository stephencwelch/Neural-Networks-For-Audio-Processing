import numpy as np
import time
from ActivationFunctions import ActivationFunction

##----------------------- General Layer Class -----------------------##
class Layer(object):
	def __init__(self,inputSize, outputSize, activationType, bias, **kwargs):
		
		self.bias = bias
		self.inputSize = inputSize
		self.outputSize = outputSize
		
		##Was epsilonInit passed in?
		if 'epsilonInit' in kwargs.keys():
			self.epsilonInit = kwargs['epsilonInit']
		else:
			self.epsilonInit = 1./(2*inputSize)

		#Create W on initialization:
		if bias:
			self.W = randInit(self.inputSize+1, self.outputSize, self.epsilonInit)
		else:
			self.W = randInit(self.inputSize, self.outputSize, self.epsilonInit)
			
		# Create Activation Function Object(s):         
		# If Activation is spline, we need to compute the total number of params within this layer. 
		# We also need to create mutliple spline instances.
		if activationType == 'spline':    
			raise Exception('Spline param counting not implemented yet - SW')
			#Create as many instance as outputs here.
			self.numParams = 0
			
		else:
			self.aF = ActivationFunction(activationType)
			self.numParams = self.W.size
	
	#Each layer has it's own parameter updater and getter:
	#Here you can pass in a matrx the same size as W, or a param vector of size (numParams)
	def setParams(self, params):
		if params.shape == (self.numParams,):
			self.W = np.reshape(params, self.W.shape)
		elif params.shape == self.W.shape:
			self.W = params
		else:
			raise Exception('params must be of size (' + str(self.numParams) + ',) or ' + str(self.W.shape) + '-SW')
			
	def getParams(self):
		return self.W.ravel()
	
	#Unravel derivatives with respect to wieghts: 
	def getDW(self):
		return self.dW.ravel()
	
	#Pass inputs through network:
	def forward(self, **kwargs):
		#Takes in optional argument x, if no arg, use previous x.
		if 'x' in kwargs.keys():
			self.x = kwargs['x']
			#Only concatente ones with new inputs!
			if self.bias:
				self.x = np.hstack((np.ones((self.x.shape[0],1)), self.x))
		
		self.z = np.dot(self.x, self.W)
		self.a = self.aF.forward(self.z)
		return self.a

##----------------------- Hidden Layer -----------------------##

#HiddenLayer Inherits from Layer:
class HiddenLayer(Layer):
	def __init__(self, inputSize=128, outputSize=4, bias=False, activationType='linear', epsilonInit = 1./256):
		
		Layer.__init__(self, inputSize=inputSize, outputSize=outputSize, activationType=activationType, \
			bias=bias, epsilonInit=epsilonInit)
		
	#Compute gradient from output to input:
	def gradAcross(self, **kwargs):
		#delta, backpropogating error- (numExamples x outputSize)
		#Only update forward pass stuff if there's a new x!
		if 'x' in kwargs.keys():
			self.forward(x=kwargs['x'])
			

		if 'delta' in kwargs.keys():
			self.delta = kwargs['delta']

		self.dOutdz = self.aF.prime(self.z)

		if self.bias:
			return np.dot((self.delta*self.dOutdz), self.W[1:,:].T)
		else:
			return np.dot((self.delta*self.dOutdz), self.W.T)

	#Gradient with respect to layer weights:
	def weightGrad(self, **kwargs):
		#Takes in optional argument x, if no arg, use previous argument.
		#delta, backpropogating error - (numExamples x outputSize)
		#x - (numExamples x inputSize)
		#Only update forward pass stuff if there's a new x!
		if 'x' in kwargs.keys():
			self.x = kwargs['x']
			self.forward(x=self.x)
			  

		if 'delta' in kwargs.keys():
			self.delta = kwargs['delta']

		self.dOutdz = self.aF.prime(self.z)		     
		self.dW = np.dot(self.x.T, self.delta*self.dOutdz)

##----------------------- Output Layer  -----------------------##

#OutputLayer Inherits from Layer:
class OutputLayer(Layer):
	def __init__(self, inputSize=4, outputSize=1, bias=False, activationType='linear', epsilonInit = 1./256):
		
		Layer.__init__(self, inputSize=inputSize, outputSize=outputSize, activationType=activationType, \
			bias=bias, epsilonInit=epsilonInit)
	
	def cost(self, **kwargs):
		#Take in new target values, y, or use existing if no argument.
		#y - (numExamples, outputSize)
		if 'x' in kwargs.keys():
			self.forward(x=kwargs['x'])
			self.dOutdz = self.aF.prime(self.z)   

		if 'y' in kwargs.keys():
			self.y = kwargs['y']

		if self.y.shape != self.a.shape:
			raise Exception('y and a must be the same shape!- SW')
		else:
			self.delta = -(self.y-self.a)

		C = 0.5*np.sum(self.delta**2)

		#Want to return a 0-dimension cost - not a single element list or array:
		if C.ndim > 0:
			C = C[0]
		
		return C
		
	#Compute gradient from output to input:
	def gradAcross(self, **kwargs):
		#Take in new target values, y, or use existing if no argument.
		#y - (numExamples, outputSize)
		if 'x' in kwargs.keys():
			self.forward(x=kwargs['x'])			

		if 'y' in kwargs.keys():
			self.y = kwargs['y']

		if self.y.shape != self.a.shape:
			raise Exception('y and a must be the same shape! -SW')
		else:
			self.delta = -(self.y-self.a)
	

		self.dOutdz = self.aF.prime(self.z)		
		if self.bias:
			return np.dot((self.delta*self.dOutdz), self.W[1:,:].T)
		else:
			return np.dot((self.delta*self.dOutdz), self.W.T)

	#Gradient with respect to layer weights:
	def weightGrad(self, **kwargs):
		#Takes in optional argument x, if no arg, use previous argument.
		#x - (numExamples x inputSize)
		#y - (numExamples, outputSize)
		if 'x' in kwargs.keys():
			self.forward(x=kwargs['x'])  

		if 'y' in kwargs.keys():
			self.y = kwargs['y']

		if self.y.shape != self.a.shape:
			raise Exception('y and a must be the same shape!')
		else:
			self.delta = -(self.y-self.a)
		

		self.dOutdz = self.aF.prime(self.z)
		self.dW = np.dot(self.x.T, self.delta*self.dOutdz)


##----------------------- Helper Functions -----------------------##
def randInit(l_in, l_out, epsilonInit):
	return np.random.rand(l_in,l_out) * 2 * epsilonInit - epsilonInit