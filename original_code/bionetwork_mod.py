import torch
import torch.nn as nn
import scipy.sparse
import numpy.random
import pandas
import numpy
#import activationFunctions
from scipy.sparse.linalg import eigs

import importlib.util
import sys
import os
lembas_path = '../LEMBAS/original_code'

def set_seeds(seed: int=888):
    """Sets random seeds for torch operations.

    Parameters
    ----------
    seed : int, optional
        seed value, by default 888
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def np_to_torch(arr: numpy.array, dtype: torch.float32, device: str = 'cpu'):
    return torch.tensor(arr, dtype=dtype, device = device)

def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

activationFunctions = import_from_path('activationFunctions', os.path.join(lembas_path, 'activationFunctions.py'))

def trainingParameters(**attributes):
    #set defaults
    params = {'targetSteps': 100, 'maxSteps': 300, 'expFactor': 20, 'leak': 0.01, 'tolerance': 1e-5}

    for curKey in params.keys():
        if curKey in attributes.keys():
            params[curKey] = attributes[curKey]

    if 'spectralTarget' in attributes.keys():
        params['spectralTarget'] = attributes['spectralTarget']
    else:
        params['spectralTarget'] = numpy.exp(numpy.log(params['tolerance'])/params['targetSteps'])

    return params


class model(torch.nn.Module):
    def __init__(self, networkList, nodeNames, modeOfAction, inputAmplitude, projectionAmplitude, inName, outName, bionetParams, seed,device, activationFunction='MML', valType=torch.float32):
        super(model, self).__init__()
        self.inputLayer = projectInput(nodeNames, inName, inputAmplitude, valType, device = device)
        self.network = bionet(networkList, len(nodeNames), modeOfAction, bionetParams, activationFunction, valType, seed = seed, device = device)
        self.projectionLayer = projectOutput(nodeNames, outName, projectionAmplitude, valType, device)
        self.projectionAmplitude = projectionAmplitude

    def forward(self, X):
        fullX = self.inputLayer(X)
        fullY = self.network(fullX)
        Yhat = self.projectionLayer(fullY)
        return Yhat, fullY
    
    def L2Reg(self, L2):
        inputL2 = self.inputLayer.L2Reg(L2)
        signalingL2 = self.network.L2Reg(L2)
        projectionL2 = self.projectionLayer.L2Reg(L2)
        return inputL2 + signalingL2 + projectionL2
    
    def applyUniformLoss(self, stateLossFactor, signalingFull):
        stateLossFactor = torch.tensor(stateLossFactor, dtype = signalingFull.dtype, device = signalingFull.device)
        #loss = stateLossFactor * uniformLossBatch(signalingFull, maxConstraintFactor = 1, targetMax = 1/self.projectionAmplitude)
        loss = stateLossFactor * alternativeUniform(signalingFull, targetMax = 1/self.projectionAmplitude)
        return loss

    def addNoiseToAllGradients(self, noiseLevel, seed):
        allParams = list(self.parameters())
        set_seeds(seed)
        for i in range(len(allParams)):
            if allParams[i].requires_grad:
                noise = torch.randn(allParams[i].grad.shape, dtype=allParams[i].dtype, device=allParams[i].device)
                allParams[i].grad += noiseLevel *noise#* torch.randn(allParams[i].grad.shape, dtype=allParams[i].dtype, device=allParams[i].device)       

    # def setDevice(self, device):
        # self.inputLayer.setDevice(device)
        # self.network.setDevice(device)
        # self.projectionLayer.setDevice(device)

class projectInput(nn.Module):
    def __init__(self, nodeList, inputNames, projectionAmplitude, dtype, device):
        super().__init__()
        self.device = device
        self.size_in = len(inputNames)
        self.size_out = len(nodeList)
        self.dtype = dtype
        dictionary = dict(zip(nodeList, list(range(len(nodeList)))))
        self.nodeOrder = torch.tensor(numpy.array([dictionary[x] for x in inputNames]), device = self.device)
        weights = projectionAmplitude * torch.ones(len(inputNames), dtype=dtype, device = self.device)
        self.weights = nn.Parameter(weights)
        self.projectionAmplitude = projectionAmplitude

    def forward(self, x):
        curIn = torch.zeros([x.shape[0],  self.size_out], dtype=self.dtype, device=self.device)
        curIn[:, self.nodeOrder] = self.weights * x
        return curIn
    
    def L2Reg(self, L2):
        projectionL2 = L2 * torch.sum(torch.square(self.weights - self.projectionAmplitude))  
        return projectionL2    



class projectOutput(nn.Module):
    def __init__(self, nodeList, outputNames, projectionAmplitude, type, device):
        super().__init__()

        self.size_in = len(nodeList)
        self.size_out = len(outputNames)
        # self.dtype = type
        # self.device = device

        dictionary = dict(zip(nodeList, list(range(len(nodeList)))))
        self.nodeOrder = torch.tensor(numpy.array([dictionary[x] for x in outputNames]), device = device)

        #bias = torch.zeros(len(outputNames), dtype=type)
        weights = projectionAmplitude * torch.ones(len(outputNames), dtype=type, device = device)
        self.weights = nn.Parameter(weights)
        self.projectionAmplitude = projectionAmplitude

    def forward(self, x):
        curOut = self.weights * x[:, self.nodeOrder]
        return curOut

    def L2Reg(self, L2):
        projectionL2 = L2 * torch.sum(torch.square(self.weights - self.projectionAmplitude))  
        return projectionL2

class bionet(nn.Module):
    def __init__(self, networkList, size, modeOfAction, parameters, activationFunction, dtype, seed, device):
        super().__init__()
        self.param = parameters
        self.seed = seed
        self.device = device

        self.size_in = size
        self.size_out = size

        self.networkList = (np_to_torch(networkList[0,:], dtype = torch.int32, device = 'cpu'), 
                          np_to_torch(networkList[1,:], dtype = torch.int32, device = 'cpu'))
        self.modeOfAction = torch.tensor(modeOfAction, dtype=torch.bool, device = self.device)
        self.dtype = dtype
        #H0 = 0.5 * torch.rand((size, 1), dtype=dtype)
        #H0 = torch.zeros((size, 1), dtype=dtype)

        # initialize weights and biases
        weightValues, bias = self.initializeWeights()
        self.mask = self.makeMask(self.networkList, size)
        weights = torch.zeros(self.mask.shape, dtype = self.dtype, device = self.device)
        weights[self.networkList] = weightValues
        
        self.modeOfActionValues, self.modeOfActionMask = self.makeMOAMask()

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        #self.H0 = nn.Parameter(H0)
             
        #self.step = forwardNetworkGPU.FFnet(networkList, size, size)
        
        if activationFunction == 'MML':
            self.activation = activationFunctions.MMLactivation
            self.delta = activationFunctions.MMLDeltaActivation
            self.oneStepDeltaActivationFactor = activationFunctions.MMLoneStepDeltaActivationFactor
        elif activationFunction == 'leakyRelu':
            self.activation = activationFunctions.leakyReLUActivation
            self.delta = activationFunctions.leakyReLUDeltaActivation
            self.oneStepDeltaActivationFactor = activationFunctions.leakyReLUoneStepDeltaActivationFactor     
        elif activationFunction == 'sigmoid':
            self.activation = activationFunctions.sigmoidActivation
            self.delta = activationFunctions.sigmoidDeltaActivation
            self.oneStepDeltaActivationFactor = activationFunctions.sigmoidOneStepDeltaActivationFactor
        else:
            print('No activation function!')
            
    def makeMOAMask(self):
        MOAsigned = self.modeOfAction[0, :].type(torch.long) - self.modeOfAction[1, :].type(torch.long) #1=activation -1=inhibition, 0=unknown
        weights = torch.zeros(self.size_out, self.size_in, dtype=torch.long, device = self.device)
        weights[self.networkList] = MOAsigned
        MOAmask = weights == 0
        return weights, MOAmask

    def makeMask(self, networkList, size):
        weights = torch.zeros(size, size, dtype=bool, device = self.device)
        weights[self.networkList] = True
        weights = torch.logical_not(weights)
        return weights
        
    def forward(self, x):
        self.applySparsity()
        tol = self.param['tolerance']
        iterations = self.param['maxSteps']
        
        condition = torch.tensor(True, device=x.device)
        transposedX = x.T
        transposedX = transposedX + self.bias
        new = torch.zeros_like(transposedX)
        
        #allSteps = []
        for i in range(iterations):
            old = new
            new = torch.mm(self.weights, new) #self.step(tmp)
            new = new + transposedX         
            new = self.activation(new, self.param['leak']) #self.step(tmp)
            
            if (i % 10 == 0) and (i > 20):
                diff = torch.max(torch.abs(new - old))    
                #diff = torch.max(torch.abs(new - allSteps[i-1]))          
                passTolerance = diff.lt(tol)
                if passTolerance == condition:
                    #allSteps.extend([new.unsqueeze(0)] * (iterations-i))
                    break
            #allSteps.append(new.unsqueeze(0))
        #allSteps = torch.cat(allSteps, axis=0)
        #allSteps = allSteps.permute([0, 2, 1])
        steadyState = new.T
        return steadyState #, allSteps
        #return bionetworkFunction.apply(x, self.weights, self.bias, self.A, self.networkList, self.param, self.activation, self.delta)


    def L2Reg(self, L2):
        #L2 = torch.tensor(L2, dtype = self.weights.dtype, device = self.weights.device)
        biasLoss = L2 * torch.sum(torch.square(self.bias))
        weightLoss = L2 * torch.sum(torch.square(self.weights))     
        #biasLoss = 0.1 * torch.sum(torch.abs(self.bias))
        #weightLoss = 0.1 * torch.sum(torch.abs(self.weights))
        return biasLoss + weightLoss

    def getWeight(self, nodeNames, source, target):
        self.A.data = self.weights.detach().numpy()
        locationSource = numpy.argwhere(numpy.isin(nodeNames, source))[0]
        locationTarget = numpy.argwhere(numpy.isin(nodeNames, target))[0]
        weight = self.A[locationTarget, locationSource][0]
        return weight
    
    
    # def getConvergenceSteps(self, YhatFull):
    #     tol = self.param['tolerance'] #half tolerance for deviation from mean
    #     deltaY = torch.abs(YhatFull[0:-1,:,:].detach() - YhatFull[1:,:,:].detach())
    #     convergenceSteps = torch.sum(torch.max(deltaY, axis=2)[0]>tol, axis=0)        
    #     return convergenceSteps
    
    # def getAproximateSR(self, YhatFull):
    #     tol = torch.tensor(self.param['tolerance'], dtype=YhatFull.dtype, device=YhatFull.device)
    #     convergenceSteps = self.getConvergenceSteps(YhatFull)
    #     aproxSpectralRadius = torch.exp(torch.log(tol)/convergenceSteps)
    #     return aproxSpectralRadius
    
    def steadyStateLoss(self, YhatFull, factor, seed, topNvalues = 10):
        factor = torch.tensor(factor, dtype=YhatFull.dtype, device=YhatFull.device)
        expFactor = torch.tensor(self.param['expFactor'], dtype=YhatFull.dtype, device=YhatFull.device)
        
        #aproxSpectralRadius = self.getAproximateSR(YhatFull.detach())
        
        

        #steps = self.param['targetSteps']
        #data = YhatFull[steps:, :, :]
        #meanData = torch.mean(data, axis=0).detach()
        #var = torch.mean(torch.square(data - meanData), axis=0)
        #var = torch.mean(torch.abs(data - meanData), axis=0)
        #mask = var.gt(tol).type(var.dtype)
        #loss = spectralRadiusFactor * factor * torch.sum(var*mask)
        #selectedValues = torch.argsort(spectralRadiusFactor, descending=True)[0:topNvalues]
        #selectedValues = 
        numpy.random.seed(seed)
        selectedValues = numpy.random.permutation(YhatFull.shape[0])[:topNvalues]
        #print(spectralRadiusFactor[selectedValues])
        #subsetYhatFull = YhatFull[:, selectedValues, :]
        deviationFromSS, aproxSpectralRadius = self.deviationFromSS(YhatFull[selectedValues,:], seed)        
        spectralRadiusFactor = torch.exp(expFactor*(aproxSpectralRadius-self.param['spectralTarget']))
        #spectralRadiusFactor = spectralRadiusFactor[selectedValues]
        
        loss = spectralRadiusFactor * deviationFromSS/torch.sum(deviationFromSS.detach())
        loss = factor * torch.sum(loss)
        aproxSpectralRadius = torch.mean(aproxSpectralRadius).item()

        return loss, aproxSpectralRadius
    
    # def deviationFromSS(self, YhatFull):
    #     nProbes = 5
    #     powerSteps = 8
        
    #     yHatSS = YhatFull[-1, :, :].detach()
    #     #factor = torch.tensor(factor, dtype=yHatSS.dtype, device=yHatSS.device)
        
    #     xPrime = self.oneStepDeltaActivationFactor(yHatSS, self.param['leak'])     
    #     xPrime = xPrime.unsqueeze(2)
        
    #     T = xPrime * self.weights
    #     Tpow = torch.pow(T, powerSteps)
    #     #Tpow2 = T * T
    #     #Tpow4 = Tpow2 * Tpow2
    #     #Tpow = Tpow4 * Tpow4
        
    #     delta0 = torch.randn((yHatSS.shape[0], yHatSS.shape[1], nProbes), dtype=YhatFull.dtype, device=YhatFull.device)
    #     deltaN = torch.matmul(Tpow, delta0)

    #     deviation = torch.sum(torch.sum(torch.square(deltaN), axis=1), axis=1)
        
    #     #loss = factor * deviation
    #     return deviation
    
    def deviationFromSS(self, yHatSS, seed, nProbes = 5, powerSteps = 50):
        
        #yHatSS = YhatFull[-1, :, :].detach()
        #factor = torch.tensor(factor, dtype=yHatSS.dtype, device=yHatSS.device)
        
        xPrime = self.oneStepDeltaActivationFactor(yHatSS, self.param['leak'])     
        xPrime = xPrime.unsqueeze(2)
        
        T = xPrime * self.weights
        set_seeds(seed)
        delta = torch.randn((yHatSS.shape[0], yHatSS.shape[1], nProbes), dtype=yHatSS.dtype, device=yHatSS.device)
        for i in range(powerSteps):
            new = delta
            delta = torch.matmul(T, new)

        deviation = torch.max(torch.abs(delta), axis=1)[0]
        aproxSpectralRadius = torch.mean(torch.exp(torch.log(deviation)/powerSteps), axis=1)
        
        deviation = torch.sum(torch.abs(delta), axis=1)
        deviation = torch.mean(torch.exp(torch.log(deviation)/powerSteps), axis=1)
        
        
        #loss = factor * deviation
        return deviation, aproxSpectralRadius    
    
    def applySparsity(self):
        self.weights.data.masked_fill_(self.mask, 0.0)
    
    def getWeights(self):
        values = self.weights[self.networkList]
        return values    

    def getViolations(self):
        #dtype = self.weights.dtype
        signMissmatch = torch.ne(torch.sign(self.weights), self.modeOfActionValues) #.type(dtype)
        signMissmatch = signMissmatch.masked_fill(self.modeOfActionMask, False)
        violations = signMissmatch[self.networkList]
        wrongSignActivation = torch.logical_and(violations, self.modeOfAction[0])
        wrongSignInhibition = torch.logical_and(violations, self.modeOfAction[1])#.type(torch.int)
        return torch.logical_or(wrongSignActivation, wrongSignInhibition)
    
    # def getNumberOfViolations(self):
    #     return torch.sum(self.getViolations(self.weights))
    
    def signRegularization(self, MoAFactor):
        MoAFactor = torch.tensor(MoAFactor, dtype = self.dtype, device = self.device)
        signMissmatch = torch.ne(torch.sign(self.weights), self.modeOfActionValues).type(self.dtype)
        signMissmatch = signMissmatch.masked_fill(self.modeOfActionMask, 0)
        loss = MoAFactor * torch.sum(torch.abs(self.weights * signMissmatch))
        return loss

    def initializeWeights(self):
        #The idea is to scale input to a source so that well connected nodes have lower weights
        #weights = 0.5 + 0.1 * (torch.rand(self.networkList.shape[1])-0.5)
        #bias = 0.01 + 0.001 * (torch.rand(self.size_in,1)-0.5)
        size = len(self.networkList[0])
        networkIn = self.networkList[0].numpy()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        weights = 0.1 + 0.1 * torch.rand(size, dtype=self.dtype, device = self.device)

        weights[self.modeOfAction[1,:]] = -weights[self.modeOfAction[1,:]]
        bias = 1e-3 * torch.ones((self.size_in, 1), dtype=self.dtype, device = self.device)
        #values, counts = np.unique(self.networkList[0,:], return_counts=True)


        for i in range(self.size_in):
            affectedIn = networkIn == i
            if numpy.any(affectedIn):
                if torch.all(weights[affectedIn]<0):
                    bias.data[i] = 1 #only affected by inhibition, relies on bias for signal

        # for i in range(self.size_in):
        #     affectedIn = self.networkList[0,:] == i
        #     fanIn = max(sum(affectedIn), 1)
        #     affectedOut = self.networkList[0,:] == i
        #     fanOut = max(sum(affectedOut), 1)
        #     weights.data[affectedIn] = weights.data[affectedIn] * numpy.sqrt(2.0/numpy.sqrt(fanIn * fanOut))

        return weights, bias

    def balanceWeights(self):
        positiveWeights = self.weights.data>0
        negativeWeights = positiveWeights==False
        positiveSum = torch.sum(self.weights.data[positiveWeights])
        negativeSum = -torch.sum(self.weights.data[negativeWeights])
        factor = positiveSum/negativeSum
        self.weights.data[negativeWeights] = factor * self.weights.data[negativeWeights]



    def preScaleWeights(self, targetRadius = 0.8):
        spectralRadius = getSR(self.weights, self.seed)
        factor = targetRadius/spectralRadius.item()
        self.weights.data = self.weights.data * factor
        # print('Pre-scaling eig')
        # optimizer = torch.optim.Adam([self.weights], lr=0.001)
        # weightFactor = self.weights
        # for i in range(1000):
        #     optimizer.zero_grad()
        #     spectralRadius = self.getSpectralRadius(weightFactor)
        #     if i % 20 == 0:
        #         print('i={:.0f}, e={:.4f}'.format(i, spectralRadius.item()))

        #     if spectralRadius.item()>targetRadius:
        #         spectralRadius.backward()
        #         optimizer.step()
        #     else:
        #         break


def getSR(weights, seed):
    A = scipy.sparse.csr_matrix(weights.detach().cpu().numpy())
    numpy.random.seed(seed)
    e1, v = eigs(A, k=1, v0 = numpy.random.rand(A.shape[0]))
    SR = numpy.abs(e1)
    return SR

def getRandomNet(networkSize, sparsity):
    network = scipy.sparse.random(networkSize, networkSize, sparsity)
    scipy.sparse.lil_matrix.setdiag(network, 0)
    networkList = scipy.sparse.find(network)
    networkList = numpy.array((networkList[1], networkList[0])) #we flip the network for rowise ordering
    nodeNames = [str(x+1) for x in range(networkSize)]
    #weights = torch.from_numpy(networkList[2])
    return networkList, nodeNames

def alternativeUniform(X, targetMin = 0, targetMax = 0.8):
    #disorderLevel = 1e-3
    targetDistribution = torch.linspace(targetMin, targetMax, X.shape[0], dtype=X.dtype, device=X.device).reshape(-1, 1)
    #targetDistribution = targetDistribution + disorderLevel * torch.randn(X.shape)
    ordered, _ = torch.sort(X, axis=0)
    distLoss = torch.sum(torch.square(ordered - targetDistribution))
    belowRange = torch.sum(X.lt(targetMin) * torch.square(X-targetMin))
    aboveRange = torch.sum(X.gt(targetMax) * torch.square(X-targetMax))
    loss = distLoss + belowRange + aboveRange    
    return loss

def loadNetwork(filename, banList = []):
    net = pandas.read_csv(filename, sep='\t', index_col=False)
    net = net[~ net["source"].isin(banList)]
    net = net[~ net["target"].isin(banList)]

    sources = list(net["source"])
    targets = list(net["target"])
    stimulation = numpy.array(net["stimulation"])
    inhibition = numpy.array(net["inhibition"])
    modeOfAction = 0.1 * numpy.ones(len(sources))  #ensuring that lack of known MOA does not imply lack of representation in scipy.sparse.find(A)
    modeOfAction[stimulation==1] = 1
    modeOfAction[inhibition==1] = -1

    networkList, nodeNames, weights = makeNetworkList(sources, targets, modeOfAction)  #0 == Target 1 == Source due to numpy sparse matrix structure
    modeOfAction = numpy.array([[weights==1],[weights==-1]]).squeeze()

    return networkList, nodeNames, modeOfAction

def makeNetworkList(sources, targets, weights):
    nodeNames = list(numpy.unique(sources + targets))
    dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
    sourceNr = numpy.array([dictionary[x] for x in sources]) #colums
    targetNr = numpy.array([dictionary[x] for x in targets]) #rows
    size = len(nodeNames)
    A = scipy.sparse.csr_matrix((weights, (sourceNr, targetNr)), shape=(size, size))
    networkList = scipy.sparse.find(A)
    weights = networkList[2]
    networkList = numpy.array((networkList[1], networkList[0]))  #0 == Target 1 == Source due to numpy sparse matrix structure
    return networkList, nodeNames, weights

def saveParam(model, nodeList, fileName):
    nodeList = numpy.array(nodeList).reshape([-1, 1])

    #Weights
    networkList = model.network.networkList
    sources = nodeList[networkList[1]]
    targets = nodeList[networkList[0]]
    paramType = numpy.array(['Weight'] * len(sources)).reshape([-1, 1])
    values = model.network.weights.detach().numpy().reshape([-1, 1])
    data1 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    #Bias
    sources = nodeList
    targets = numpy.array([''] * len(sources)).reshape([-1, 1])
    paramType = numpy.array(['Bias'] * len(sources)).reshape([-1, 1])
    values = model.network.bias.detach().numpy().reshape([-1, 1])
    data2 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    #Projection
    projectionList = model.projectionLayer.nodeOrder
    sources = nodeList[projectionList]
    targets = numpy.array([''] * len(sources)).reshape([-1, 1])
    paramType = numpy.array(['Projection'] * len(sources)).reshape([-1, 1])
    values = model.projectionLayer.weights.detach().numpy().reshape([-1, 1])
    data3 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    #Input projection
    projectionList = model.inputLayer.nodeOrder
    sources = nodeList[projectionList]
    targets = numpy.array([''] * len(sources)).reshape([-1, 1])
    paramType = numpy.array(['Input'] * len(sources)).reshape([-1, 1])
    values = model.inputLayer.weights.detach().numpy().reshape([-1, 1])
    data4 = numpy.concatenate((sources, targets, paramType, values), axis=1)

    data = numpy.concatenate((data1, data2, data3, data4))

    pd = pandas.DataFrame(data)
    pd.columns = ['Source', 'Target', 'Type', 'Value']
    pd.to_csv(fileName, sep = '\t', quoting = None, index = False)

def loadParam(fileName, model, nodeNames):
    dictionary = dict(zip(nodeNames, list(range(len(nodeNames)))))
    data = pandas.read_csv(fileName, delimiter = '\t')

    #Reset model to zero
    model.inputLayer.weights.data = torch.zeros(model.inputLayer.weights.shape)
    model.network.weights.data = torch.zeros(model.network.weights.shape)
    model.network.bias.data = torch.zeros(model.network.bias.shape)
    model.projectionLayer.weights.data = torch.zeros(model.projectionLayer.weights.shape)

    inputLookup = model.inputLayer.nodeOrder
    projectionLookup = model.projectionLayer.nodeOrder

    for i in range(data.shape[0]):
        curRow = data.iloc[i,:]
        source = dictionary[curRow['Source']]
        value = curRow['Value']
        if curRow['Type'] == 'Weight':
            target = dictionary[curRow['Target']]
            model.network.weights.data[target, source] = value
        elif curRow['Type'] == 'Bias':
            model.network.bias.data[source] = value
        elif curRow['Type'] == 'Projection':
            model.projectionLayer.weights.data[projectionLookup == source] = value
        elif curRow['Type'] == 'Input':
            model.inputLayer.weights.data[inputLookup == source] = value

    return model


def getMeanLoss(criterion, Y):
    averagePerOutput = torch.mean(Y, dim=0)
    medianPerOutput = torch.from_numpy(numpy.median(Y, axis=0))
    errorFromPredictingTheMean = criterion(Y, averagePerOutput)
    errorFromPredictingTheMedian = criterion(Y, medianPerOutput)
    return (errorFromPredictingTheMean.item(), errorFromPredictingTheMedian.item())


def generateConditionNames(X, inName):
    inName = numpy.array(inName)
    X = X.detach().numpy()
    names = numpy.empty(X.shape[0], dtype=object)
    for i in range(len(names)):
        curSelection = X[i,:]>0
        if sum(curSelection) == 0:
            names[i] = '(none)'
        else:
            curLigands = list(inName[curSelection])
            names[i] = '_'.join(curLigands)
    return names



def generateRandomInput(model, N, simultaniousInput):
    sizeX = model.inputLayer.weights.shape[0]
    X = torch.zeros((N, sizeX), dtype=torch.double)
    for i in range(1, N): #leave first block blank
        selected = numpy.random.randint(sizeX, size=simultaniousInput)
        X[i, selected] = torch.rand(simultaniousInput, dtype=torch.double)
    return X

def oneCycle(e, maxIter, maxHeight = 1e-3, startHeight=1e-5, endHeight=1e-5, peak = 1000):
    phaseLength = 0.95 * maxIter
    if e<=peak:
        effectiveE = e/peak
        lr = (maxHeight-startHeight) * 0.5 * (numpy.cos(numpy.pi*(effectiveE+1))+1) + startHeight
    elif e<=phaseLength:
        effectiveE = (e-peak)/(phaseLength-peak)
        lr = (maxHeight-endHeight) * 0.5 * (numpy.cos(numpy.pi*(effectiveE+2))+1) + endHeight
    else:
        lr = endHeight
    return lr
