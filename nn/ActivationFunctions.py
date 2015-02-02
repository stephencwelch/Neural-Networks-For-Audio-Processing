##-----------------------Activation Functions -----------------------##
import numpy as np

class LinearActivation(object):
	#Activation functions now have a state, which is stored values for z. 
	def __init__(self):
		self.z = 0
	
	def forward(self, z):
		self.z = z
		return z
	
	#Prime can take one or zeros arguments, if no argument is given, it will use the stored value for z.
	def prime(self, *args):
		if len(args) > 0:
			self.z = args[0]
			
		return np.ones(self.z.shape)
	
class TansigActivation(object):
	#Activation functions now have a state, which is stored values for z. 
	def __init__(self):
		self.z = 0
	
	def forward(self, z):
		self.z = z
		return  float(2)/(1+np.exp(-2*self.z))-1
	
	#Prime can take one or zeros arguments, if no argument is given, it will use the stored value for z.
	def prime(self, *args):
		if len(args) > 0:
			self.z = args[0]
			
		e = np.exp(2*self.z)
		return 4*e/((1+e)**2)
	
class SigmoidActivation(object):
	#Activation functions now have a state, which is stored values for z. 
	def __init__(self):
		self.z = 0
	
	def forward(self, z):
		self.z = z
		return 1/(1+np.exp(-self.z))
	
	#Prime can take one or zeros arguments, if no argument is given, it will use the stored value for z.
	def prime(self, *args):
		if len(args) > 0:
			self.z = args[0]
			
		e = 1/(1+np.exp(-self.z))
		return (1-e)*e
	   
class ActivationFunction(object):
#Options: Linear, sigmoid, tansig, spline.

	def __init__(self, type='linear'):
		self.type = type
		
		#If spline, set hasParams flag to True:
		if self.type == 'spline':
			self.hasParams = True
			#but this has not been impelemented yet:
			raise Exception('Spline has not been implemented yet! -SW')
			
		#Linear
		elif self.type == 'linear':
			self.hasParams = False
			
			#Create activation object to do processing:
			la = LinearActivation()
			self.forward = la.forward
			self.prime = la.prime
			
		#Tansig
		elif self.type == 'tansig':
			self.hasParams = False
			
			#Create activation object to do processing:
			ta = TansigActivation()
			self.forward = ta.forward
			self.prime = ta.prime
		
		#Sigmoid
		elif self.type == 'sigmoid':
			self.hasParams = False
			
			#Create activation object to do processing:
			sa = SigmoidActivation()
			self.forward = sa.forward
			self.prime = sa.prime

		else:
			raise Exception('Type must be linear, tansig, sigmoid, or spline. -SW')