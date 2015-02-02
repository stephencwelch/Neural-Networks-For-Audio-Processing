import numpy as np
from scipy import fft
import cPickle as pickle
import sys
import time

#Data class is a nice way to load and keep track of training and testing data.
#Stephen Welch 

class Data(object):
    def __init__(self, pickleName, exampleSize=128, target='mic1'):
        pickleFileName = "Pickles/" + pickleName + ".pickle"
        pickleFile = open(pickleFileName, 'rb')
        alignedSignals = pickle.load(pickleFile)
        pickleFile.close()

        self.targetSignal = alignedSignals[target]
        self.piezoAudio = alignedSignals['piezo']

        metaData = alignedSignals['metaData']

        #add metaData to Data Class Object:
        for name in metaData.keys():
            setattr(self, name, metaData[name])
        
        #Compute whole designMatrix, require examples size on instantiation, 
        #so we only compute design matrix once:
        self.exampleSize =  exampleSize
        self.overallDesignMatrix = self.makeToeplitz(self.piezoAudio, self.exampleSize)

     
    def sample(self, indices):
        X = self.overallDesignMatrix[indices,:]
        y = self.targetSignal[indices + self.exampleSize].reshape(len(indices),1)
        return X, y
    
    def makeToeplitz(self, inputVec, numColumns):
        #Make Toeplitz Design Matrix
        toeplitzMatrix = np.zeros((len(inputVec)-numColumns+1, numColumns))
        for k in range(numColumns):
            toeplitzMatrix[:, k] = inputVec[k:(len(inputVec)-numColumns+k+1)]
        return toeplitzMatrix
    
    def _checkNumExamples(self,numExamples):
        #Make sure we've precomputed this number of examples.
        matchFlag = 0
        for numExamplesPC in self.numTrainingExamplesList:
            if numExamplesPC==numExamples:
                matchFlag=1
                
        if matchFlag ==0:
            print "Have not precomputed random indices for " + str(numExamples) + " examples." + \
            "Please check out Data Import and Randomization.ipynb or use a number from this list:"
            print self.numTrainingExamplesList
            
    #Sample single random time slices, no previous data.       
    def simpleSample(self, numExamples=1024, randomSampleNum=0):
        self.numExamples = numExamples
        
        self.trainingIndices = self.randomIndices[str(numExamples) + 'Train'][:,randomSampleNum]
        self.testingIndices = self.randomIndices[str(numExamples) + 'Test'] [:,randomSampleNum]
        
        train = self.sample(self.trainingIndices)
        test = self.sample(self.testingIndices)
        
        return train, test
    
    #Sample Continuous Blocks, no previous data.
    def sampleContinuousBlocks(self, numBlocks=32, blockLength=128, randomSampleNum=0):
        self.numExamples=numBlocks*blockLength
        
        self.numBlocks = numBlocks
        self.blockLength = blockLength
        
        trainingIndicesStart = self.randomIndices[str(numBlocks) + 'Train'][:,randomSampleNum]
        testingIndicesStart = self.randomIndices[str(numBlocks) + 'Test'] [:,randomSampleNum]
        
        trainingIndices = np.array([],dtype='int')
        testingIndices = np.array([],dtype='int')
        
        #Using sampled indices as starting points, I need a continuous window of blockLength:
        for blockNum in range(numBlocks):
            trainingIndices = np.concatenate((trainingIndices, np.arange(trainingIndicesStart[blockNum], \
                                                                          trainingIndicesStart[blockNum] + blockLength)))
            testingIndices = np.concatenate((testingIndices, np.arange(testingIndicesStart[blockNum], \
                                                                          testingIndicesStart[blockNum] + blockLength)))
            
        self.trainingIndices = trainingIndices
        self.testingIndices = testingIndices
        
        train = self.sample(self.trainingIndices)
        test = self.sample(self.testingIndices)
        
        return train, test

    def sampleForPlayBack(self, startTime=1, endTime=6):

        indices = np.arange(startTime*self.Fs, endTime*self.Fs)

        X = self.overallDesignMatrix[indices,:]
        y = self.targetSignal[indices + self.exampleSize].reshape(indices.shape[0],1)
        rawPiezo = self.piezoAudio[indices + self.exampleSize].reshape(indices.shape[0],1)
        return X, y, rawPiezo






