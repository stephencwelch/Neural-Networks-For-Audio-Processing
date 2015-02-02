from scipy import optimize
import numpy as np
import time
from matplotlib import pyplot

class trainer(object):
    def __init__(self, N):
        #N is our neural network object: 
        self.N = N
        self.maxIter = self.N.hP.maxIter
        self.optimizationMethod = self.N.hP.optimizationMethod
        
        #List of error throughout training:
        self.JTrain = []
        self.JTest = []
        
    def checkPairSize(self, X, y):
        #Make sure dimensions are correct for pair:
        numInputExamples = X.shape[0]
        numOutputExamples = y.shape[0]
        
        if (X.shape != (numOutputExamples, self.N.inputSize)) or (y.shape != (numInputExamples, 1)):
            raise Exception('X must be (numExamples, inputSize), y must be (numExamples, 1) - SW')
        else:
            return numInputExamples
        
    #Helper function to track errors during training:
    def callBackF(self, params):
        self.JTrain.append(self.N.costFunction(params, self.trainX, self.trainY))
        
        #Validation Mode?
        if self.validationMode:
            self.JTest.append(self.N.costFunction(params, self.testX, self.testY))
        
        
    def train(self, trainX, trainY, **kwargs):
    
        ## Takes at least: trainX, trainY, 
        ## may take: testX, testY for validation mode
        ## or params0, starting parameters. 
        
        startTime = time.clock()
        print 'training...'
        self.trainX = trainX
        self.trainY = trainY
        
        #Check dimensionality: 
        self.numTrainingExamples = self.checkPairSize(self.trainX, self.trainY)
             
        #Do we have testing data?
        if 'testX' in kwargs:
            self.testX = kwargs['testX']
            self.testY = kwargs['testY']
            self.numTestingExamples = self.checkPairSize(self.testX, self.testY)
            self.validationMode = True
        else:
            self.validationMode = False
        
        #Check if starting params are passed it:
        if 'params0' in kwargs:
            self.params0 = kwargs['params0']
            #Pass starting params into netowrk:
            N.params.unpack(self.params0)
        else:
            #Grab current params for starting point:
            self.params0 = self.N.params.pack()
            
        #Reset cost lists:
        self.JTrain = []
        self.JTest = []
        
        ##Optimization time:
        options = {'maxiter':self.maxIter, 'disp': True}
        
        self.optimizationResults = optimize.minimize(self.N.costFunctionPrime, self.params0, jac=True, \
                                                 method=self.optimizationMethod, args = (self.trainX, self.trainY), 
                                                 options=options, callback = self.callBackF)
        
        ##Pass resulting, tuned params back into N class:
        self.N.params.unpack(self.optimizationResults.x)
        
        self.dateTimeTrained = time.strftime("%d-%m-%Y %H.%M.%S")
        self.timeToTrain = str(time.clock()-startTime)
        
        #If optmize progresses, compute error:
        if len(self.JTrain)> 0:
            self.trainError = 100*float(self.JTrain[-1])/self.numTrainingExamples

            if self.validationMode:
                self.testError = 100*float(self.JTest[-1])/self.numTestingExamples
                self.percentOverfit = 100*((self.testError-self.trainError)/self.trainError)

        print 'Done! time elapsed = ' + self.timeToTrain + 's.'
        
        
    #i'll need to import pyplot or something for this to work ultimately, look @ vizualizeGradients: 
    def plot(self):
        pyplot.plot(100*np.array(self.JTrain)/self.numTrainingExamples, linewidth=2)
        print 'Training MSE: ' + str(round(self.trainError, 5))
        
        if self.validationMode:
            pyplot.plot(100*np.array(self.JTest)/self.numTestingExamples, linewidth =2)
            
            pyplot.legend(['Train', 'Test'])
            print 'Testing MSE: ' + str(round(self.testError, 5))
            print 'Overfit: ' + str(round(self.percentOverfit, 3)) + '%'
            
        pyplot.grid(1)    
        pyplot.title('Cost')
       
        