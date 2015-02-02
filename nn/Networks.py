import numpy as np
import Layers

## ------------------- Hyper Parameters Class -------------------- ##

class HyperParameters(object):
    def __init__(self, **kwargs):
        #Optional Agruments - epsilonInit, numHiddenLayers, layerSizes, acivations, maxIter
        
        #Dict of defualts:
        defaults = {}
        defaults['epsilonInit'] = 1./256
        defaults['numHiddenLayers'] = 1
        defaults['layerSizes'] = [128,4,1]
        defaults['activations'] = ['tansig', 'tansig']
        defaults['maxIter'] = 100
        defaults['optimizationMethod'] = 'CG'
        
        for name in kwargs.keys():
            setattr(self, name, kwargs[name])

        #If args are passed in, used these, if not, use defaults:
        for defaultName in defaults.keys():
            if defaultName in kwargs:
                setattr(self, defaultName, kwargs[defaultName])
            else:
                setattr(self, defaultName, defaults[defaultName])

## ------------------- Params Class -------------------- ##

class Params(object):
    def __init__(self,listOfLayers):
        numLayers = len(listOfLayers)
        
        self.listOfLayers = listOfLayers
        self.layerSizes = np.zeros(numLayers, dtype='int')
        
        for layerNum in range(numLayers):
            self.layerSizes[layerNum] = listOfLayers[layerNum].numParams
            
        self.paramVectorTransitionPoints = np.concatenate(([0], np.cumsum(self.layerSizes)))
        
        self.length = self.paramVectorTransitionPoints[-1]
            
    def unpack(self, params):
        #unpacks param vector and sends it into each layer object. 
        if params.shape[0] != self.length:
            raise Exception('params must be of length ' + str(self.length) + '! -SW')

        for layerNum, layer in enumerate(self.listOfLayers):
            layer.setParams(params[self.paramVectorTransitionPoints[layerNum]:self.paramVectorTransitionPoints[layerNum+1]])
            
    def pack(self):
        #Grabs and returns params from each layer object
        params = np.array([])
        for layerNum, layer in enumerate(self.listOfLayers):
            params = np.concatenate((params, layer.getParams()))
            
        return params
    
    def packGrads(self):
        grads = np.array([])
        for layer in self.listOfLayers:
            grads = np.concatenate((grads, layer.getDW()))
            
        return grads
            
## ------------------- Simple One Layer Temporal Error Network -------------------- ##

class One_Layer_Network(object):
    def __init__(self, hyperParams):
        self.hP = hyperParams
        
        #Instantiate Layers:
        self.hL = Layers.HiddenLayer(inputSize = self.hP.layerSizes[0], outputSize = self.hP.layerSizes[1], \
                                  activationType = self.hP.activations[0])
        self.oL = Layers.OutputLayer(inputSize = self.hP.layerSizes[1], outputSize = self.hP.layerSizes[2], \
                                  activationType = self.hP.activations[1])
        
        self.inputSize = self.hL.inputSize
        
        #Initialize Params
        self.params = Params([self.hL, self.oL])
        
    def forward(self, x):
        self.a2 = self.hL.forward(x=x)
        self.a3 = self.oL.forward(x=self.a2)
        return self.a3
        
    def costFunction(self, paramVals, x, y):        
        #Distribute new params:
        self.params.unpack(paramVals)
        
        yHat = self.forward(x=x)
        return self.oL.cost(y=y)
        
    def costFunctionPrime(self, paramVals, x, y):
        #Compute cost:
        J = self.costFunction(paramVals, x, y)
        
        #Backpropogation:
        delta3 = self.oL.gradAcross(y=y)
        self.oL.weightGrad(y=y)
        self.hL.weightGrad(delta=delta3)
        
        #Pack Compute Gradients:
        grad = self.params.packGrads()
        
        return (J, grad)



