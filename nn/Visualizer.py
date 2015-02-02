from matplotlib import pyplot
import numpy as np
import time

class Visualizer(object):
	def __init__(self, N):
		self.N=N

	def vizualizeGradients(self, X, y, **kwargs):
		#Optional Arguments: params, paramsIndicesToVisualize
		if 'params' in kwargs.keys():
			setattr(self, 'params', kwargs('params'))
		else:
			self.params = self.N.params.pack()

		if 'paramsIndicesToVisualize' in kwargs.keys():
			setattr(self, 'paramsIndicesToVisualize', kwargs('paramsIndicesToVisualize'))
		else:
			#Just pick randomly!
			randVec = np.arange(len(self.params))
			np.random.shuffle(randVec)
			self.paramsIndicesToVisualize = randVec[0:5]

		neighborhoodSize = 15

		costs = np.zeros((200,len(self.paramsIndicesToVisualize)))
		grads = np.zeros((200,len(self.paramsIndicesToVisualize)))
		neighborhoods = np.zeros((200,len(self.paramsIndicesToVisualize)))

		startTime = time.clock()
		#And now I'll examing the neighborhoods of the params by brute force:
		for i, paramIndex in enumerate(self.paramsIndicesToVisualize):
			neighborhood = np.linspace(self.params[paramIndex]-neighborhoodSize, self.params[paramIndex]+neighborhoodSize, 200)

			for j, paramVal in enumerate(neighborhood):
				thetaMod = np.copy(self.params)
				thetaMod[paramIndex] = paramVal
				# For some reason I think this calling the wrong costFunctionBatch, so I added the "self"
				cost, grad = self.N.costFunctionPrime(thetaMod, X, y)
				costs[j , i] = cost
				grads[j , i] = grad[paramIndex]

			neighborhoods[:,i] = neighborhood

		stepSize = neighborhood[1]-neighborhood[0]
		print 'Time Elapsed = ' + str(time.clock()-startTime) + 's. '

		fig = pyplot.figure(0,(12,12))

		for paramIndex in range(5):
			fig.add_subplot(2,3,paramIndex+1)
			#Da hood
			pyplot.plot(neighborhoods[:,paramIndex], costs[:,paramIndex])

			#Current param:
			xCurrent = self.params[self.paramsIndicesToVisualize[paramIndex]]
			yCurrent = costs[100,paramIndex]
			pyplot.scatter(xCurrent, yCurrent)
			pyplot.grid(True)

			#Da gradient
			x1 = xCurrent - 50*stepSize
			x2 = xCurrent + 50*stepSize

			gradCurrent = grads[100,paramIndex]

			y1 = yCurrent - gradCurrent*(xCurrent-x1)
			y2 = yCurrent + gradCurrent*(x2-xCurrent)
			pyplot.plot([x1,x2], [y1,y2])