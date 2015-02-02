import numpy as np

class UnitTesting(object):
    def __init__(self):
        #epsilon for numgrad computations:
        self.e = 1e-4

    ## ---------------------------- Helper Functions ------------------------------- ##
        
    def computeNumgrad(self, x, wrapper):
        perturb = np.zeros(x.shape)
        numgrad = np.zeros(x.shape)

        if x.ndim == 2:
            for n in range(x.shape[0]):
                for m in range(x.shape[1]):
                    perturb[n,m] = self.e
                    ng = (wrapper(x+perturb) - wrapper(x-perturb))/(2*self.e)

                    numgrad[n,m] = np.sum(ng)
                    perturb[n,m] = 0

        elif x.ndim == 1:
            for n in range(x.shape[0]):
                perturb[n] = self.e
                ng = (wrapper(x+perturb) - wrapper(x-perturb))/(2*self.e)     

                numgrad[n] = np.sum(ng)
                perturb[n] = 0

        return numgrad    

    def testGradient(self, wrapper, wrapperPrime, x):
        #Check if yPrime has the correct dimensions:
        yPrime = wrapperPrime(x)
        if yPrime.shape != x.shape:
            raise Exception('Gradient must be same shape as input! -SW')

        self.numgrad = self.computeNumgrad(x, wrapper)
        self.grad = wrapperPrime(x)

        return np.linalg.norm(self.grad-self.numgrad)/np.linalg.norm(self.grad+self.numgrad)
    
    ## ---------------------------- Activation Function Testing ------------------------------- ##
        
    def testActivationFunction(self, aF):
        #Setup Wrappers:
        def wrapper(x):
            y = aF.forward(x)
            return y

        def wrapperPrime(x):
            y = aF.prime(x)
            return y

        #Test wtih Scalar, Vector, and Matrix:
        print 'Testing, values should by very small:'

        err = self.testGradient(wrapper, wrapperPrime, np.random.rand(1))
        print 'Tested with Scalar: ' + str(err)

        err = self.testGradient(wrapper, wrapperPrime, np.random.rand(1,5))
        print 'Tested with Vector: ' + str(err)

        err = self.testGradient(wrapper, wrapperPrime, np.random.rand(5,5))
        print 'Tested with Matrix: ' + str(err)
        
    ## ---------------------------- LayerTesting ------------------------------- ##

    def testHiddenLayer(self, layer):

         ## ----------------- Now Test with respect to inputs (gradAccross)----------------- ##
        #Test grad across unit first:
        #Setup Wrappers:
        def wrapper(x):
            y = layer.forward(x=x)
            return y

        def wrapperPrime(x):
            #Input x values into object:
            #y = layer.forward(x=x)

            #Hardcode output grad, really important that these are all the same:
            delta = np.ones((x.shape[0], layer.outputSize))
            y = layer.gradAcross(x=x, delta=delta)
            return y

        #Test wtih Single and Multiple examples:s
        print 'Testing gradAcross, values should by very small:'

        numExamples = 1
        x = np.random.randn(numExamples, layer.inputSize)
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Tested with One Example: ' + str(err)

        numExamples = 10
        x = np.random.randn(numExamples, layer.inputSize)
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Tested with Ten Examples: ' + str(err)

        ## ----------------- Now Test with respect to weights (weightGrad) ---------------- ##
        numExamples = 1
        inputData = np.random.randn(numExamples, layer.inputSize)
        #Set x and z in layer:
        y = layer.forward(x=inputData)

        def wrapper(x): 
            layer.setParams(x)  
            y = layer.forward()   
            return y 

        def wrapperPrime(x):
            layer.setParams(x)
            #Hardcode output grad, really important that these are all the same:
            layer.forward()
            delta = np.ones((layer.x.shape[0], layer.outputSize))
            layer.weightGrad(delta=delta)
            y = layer.getDW()
            return y

        #Test wtih Single and Multiple examples:s
        print 'Testing wieghtGrad, values should by very small:'

        x = np.random.randn(layer.getParams().shape[0])
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Tested with One Example: ' + str(err)
        
        numExamples = 10
        inputData = np.random.randn(numExamples, layer.inputSize)
        #Set x and z in layer:
        y = layer.forward(x=inputData)

        x = np.random.randn(layer.getParams().shape[0])
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Tested with Ten Examples: ' + str(err)
        
        # ----------------- TEST OUTPUT LAYER ----------------- ##
        
    def testOutputLayer(self, layer):

        numExamples = 1
        x = np.random.randn(numExamples, layer.inputSize)
        target = np.random.randn(numExamples, 1)
        yHat = layer.forward(x=x)
        #Set y:
        y = layer.gradAcross(y=target)

        #Test grad across unit first:
        #Setup Wrappers:
        def wrapper(x):
            yHat = layer.forward(x=x)
            y = layer.cost()
            return y

        def wrapperPrime(x):
            #Input x values into object:
            yHat = layer.forward(x=x)
            y = layer.gradAcross()
            return y

        x = np.random.randn(numExamples, layer.inputSize)
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Testing gradAccros, values should by very small:'
        print 'Tested with One Example: ' + str(err)
        
        numExamples = 10
        x = np.random.randn(numExamples, layer.inputSize)
        target = np.random.randn(numExamples, 1)
        yHat = layer.forward(x=x)
        #Set y:
        y = layer.gradAcross(y=target)

        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Tested with Ten Examples: ' + str(err)
        

        ## ------------------ Test weightGrad ------------------ ##

        def wrapper(x): 
            layer.setParams(x)
            yHat = layer.forward()
            y = layer.cost()
            return y

        def wrapperPrime(x):
            layer.setParams(x)
            #This bug held me up for a while, have to call forward first to update z!
            yHat = layer.forward()
            layer.weightGrad()
            y = layer.getDW()
            return y

        numExamples = 1
        inputData = np.random.randn(numExamples, layer.inputSize)
        target = np.random.randn(numExamples, 1)
        yhat = layer.forward(x =inputData)
        y = layer.weightGrad(y=target)


        x = np.random.randn(layer.getParams().shape[0])
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Testing weightGrad, values should by very small:'
        print 'Tested with One Example: ' + str(err)

        numExamples = 10
        inputData = np.random.randn(numExamples, layer.inputSize)
        target = np.random.randn(numExamples, 1)
        yhat = layer.forward(x =inputData)
        y = layer.weightGrad(y=target)


        x = np.random.randn(layer.getParams().shape[0])
        err = self.testGradient(wrapper, wrapperPrime, x)
        print 'Tested with Ten Examples: ' + str(err)

    ## --------------------- Test Network ----------------------- ##
    #Going to break my protocol here a little, this code could be much cleaner:

    def testNetwork(self, N):
        numExamples = 1
        x = np.random.randn(numExamples, N.inputSize)
        y = np.random.rand(numExamples, N.oL.outputSize)

        paramVals = N.params.pack()
        perturb = np.zeros(paramVals.shape)
        numgrad = np.zeros(paramVals.shape)

        e = 1e-4

        for n in range(paramVals.shape[0]):
            perturb[n] = e
            numgrad[n]  = (N.costFunctionPrime(paramVals+perturb, x, y)[0] - N.costFunctionPrime(paramVals-perturb, x, y)[0])/(2*e)     
            perturb[n] = 0
            
        grad = N.costFunctionPrime(paramVals, x, y)[1]

        err = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
        print 'Testing Network, values should by very small:'
        print ('For 1 examples: ' + str(err))

        numExamples = 10
        x = np.random.randn(numExamples, N.hL.inputSize)
        y = np.random.rand(numExamples, N.oL.outputSize)

        paramVals = N.params.pack()
        perturb = np.zeros(paramVals.shape)
        numgrad = np.zeros(paramVals.shape)

        e = 1e-4

        for n in range(paramVals.shape[0]):
            perturb[n] = e
            numgrad[n]  = (N.costFunctionPrime(paramVals+perturb, x, y)[0] - N.costFunctionPrime(paramVals-perturb, x, y)[0])/(2*e)     
            perturb[n] = 0
            
        grad = N.costFunctionPrime(paramVals, x, y)[1]

        err = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)
        print ('For 10 examples: ' + str(err))


  