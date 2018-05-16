# -*- coding: utf-8 -*-
"""
Created on Wed May  9 18:04:58 2018

@author: fanta
"""

from goTime import  *
from winnerMatrix import *
from entropy import *
import matplotlib.pyplot as plt
import GPy, numpy as np

		
def compute_s_i_t(res_i, t, winner):
	std_i = []
	for i in range(0, t):
		si = res_i[i] - winner
		std_i.append(si**2)
	print("\n Std_i: " + str(std_i))
	return std_i
	
def compute_s_t(normlizedLamList_d, normalizedLamList_e, winMatrix):
	std = []
	
	for i in range(0, 204):
		result = makeTemporalPosteriorSequence( 74.0/204,  normalizedLamList_d[i],  normalizedLamList_e[i])
		print("\n Len result: " + str(len(result)))
		std_i = compute_s_i_t(result, len(result), winMatrix[i])
		std.append(std_i)	
	print("\n Std Length: " + str(len(std)))
	std_average = [float(sum(col))/len(col) for col in zip(*std)]
	#std_average = [x ** 0.5 for x in std_average]
	print("\n Average Std square root: " + str(std_average))
	
	plt.figure()
	plt.plot(np.arange(len(result)), std_average, linewidth=1.5)
	plt.xlabel("Time (t)")
	plt.ylabel("S_t")
	plt.title("Residual filtration posterior volatility over time")
	plt.xlim(0, len(result))
	plt.ylim(0,0.25)
	
	

def makeStopEvent(seqs, winMatrix):
	pedWinEvents = [27, 10, 48, 61, 45, 16, 41, 40, 31]
	carWinEvents = [9, 4, 12, 29, 8, 38, 19, 7, 21, 13]	
	
	for i in range(0, len(seqs)):
		#print("i: " + str(i))
		visit = False
		
		if (winMatrix[i] == 1):
			for stop in pedWinEvents:
				for j in range(0, len(seqs[i])):
					#print("j: " + str(j))
					if (seqs[i][j] == stop and visit == False):
						#seqs[i].insert(j+1, 62)
						tmp = []
						for k in range(j, 26):
							tmp.append(62)
						seqs[i] = seqs[i][0:j+1] + tmp 
						visit = True
							
		else:
			for j in range(0, len(seqs[i])):
				for stop in carWinEvents:
					#print("j: " + str(j))
					if ( seqs[i][j] == stop and visit == False):
						tmp = []
						for k in range(j, 26):
							tmp.append(63)
						seqs[i] = seqs[i][0:j+1] + tmp 
						visit = True
	
	# hack for empty sequences
	tmp = []
	for k in range(0, 27):
		tmp.append(63)	
	seqs[67] = tmp	
	
	# hack for empty sequence
	tmp = []
	for k in range(0, 27):
		tmp.append(62)	
	seqs[198] = tmp

							
	return seqs

# to deal with zeros for log funcion
# add 1 to all the frequencies
def goodTuring(lam):
		
	for freq in lam:	
		lam[freq] = (lam[freq][0] + 1, lam[freq][1] + 1)
		
	return lam


# compute frequency/likelihood (p(d_i|w), p(d_i|L)) for the descriptor of one interaction
# lDesc: list of 12 descriptors (age, distraction, gender...)
# descriptor: list of descriptors for one interaction
# winner: 0 for vehicle winning and 1 for pedestrian winning
# plam_d: list of frequencies/likelihood for all descriptors as a tuple (p(d_i|w), p(d_i|L))
def makeFreqLams_d(lDesc, descriptor, winner, lam_d):
	
	for j in range(0, len(lDesc)):
	
		if any( lDesc[j] in s for s in descriptor): # single interacting car
			if(j<10 and winner == 1):
				lam_d[j] = (lam_d[j][0] + 1, lam_d[j][1])  # lam_d[j][0] = lam_d_given_W and lam_d[j][1] = lam_d_given_L
			elif(j< 10 and winner == 0):
				lam_d[j] = (lam_d[j][0], lam_d[j][1] + 1)  # lam_d[j][0] = lam_d_given_W and lam_d[j][1] = lam_d_given_L
			elif(j == 10):
				if any( "Distraction_None" in s for s in descriptor): 
					print("Distraction none")
				elif(winner == 1):
					#print(descriptor)
					lam_d[j] = (lam_d[j][0] + 1, lam_d[j][1])
				else:
					lam_d[j] = (lam_d[j][0], lam_d[j][1] + 1)
			elif(j >= 11):
				if(winner == 1):
					lam_d[11] = (lam_d[11][0] + 1, lam_d[11][1])
				else:
					lam_d[11] = (lam_d[11][0], lam_d[11][1] + 1)
	return lam_d
						
# compute not frequency/likelihood of each sequence (p(f_i|W), p(f_i|L)) and descriptors (p(d_i|w), p(d_i|L))
# seqs: list of sequences
# descriptorss: list of all descriptors
# dct_reverse: dict of 62 sequence labels
# winMatrix: matrix of the winner for each interaction
def makeNotNormalizedFreqs_d(seqs, descriptor, winMatrix, lDesc):

	freqs_d = dict()
	
	for i in range(12):
		freqs_d[i] = (0,0)
	
	for i in range(0, len(seqs)):
		winner = winMatrix[i]
		freqs_d = makeFreqLams_d(lDesc, descriptorss[i], winner, freqs_d)  # lam_d: list of likelihood for descriptors as (p(d_i|w), p(d_i|L))
				
	#print("freqs_d: " + str(freqs_d))
						
	return freqs_d   
	

def makeNotNormalizedFreq_e(seqs, dct_reverse, winMatrix):
	
	freqs_e = dict()
		
	for i in range(0, len(dct_reverse)):
		freqs_e[i] = (0,0)
	
	for i in range(0, len(seqs)):
		winner = winMatrix[i]

		for label in seqs[i]:
			if(winner == 1):
				freqs_e[label] = (freqs_e[label][0] + 1, freqs_e[label][1])
			else:
				freqs_e[label] = (freqs_e[label][0], freqs_e[label][1] + 1)   # lam_f: list of for sequence features (p(f_i|W), p(f_i|L))
	
	#print("freqs_e: " + str(freqs_e))
						
	return freqs_e

#assume there are two types of feature: descriptors (which occur all at t=0) and events (which occur at t)

#bayesian fusion of two bernoulli probs (my favourite function of all time)
def fuseProbs(p1, p2):
    return (p1*p2)/((p1*p2)+((1-p1)*(1-p2)))
				

def makeNormalizedLikelihood( p_di_given_W , p_di_given_L):
	
	return p_di_given_W/(p_di_given_W + p_di_given_L + 0.0)

#all of this is for one single interaction
def makeTemporalPosteriorSequence( pi_W,  lams_d,  lams_e):   
	#where lam_e is a 1*F list of normalized likelihoods for each events in sequence
	#lam_d is 1*D list of normalized likelihoods for each descriptors 
	#eg.   [0.2 , 0.6, 0.4, 0.9 ] each is one descriptor which occurred, and shows its normalized lik
	#return: temporal sequence over t of posteriors P(W|D_{t=1},  E_{2:t} )  (t=0 is just prior)
    
	result = []    #to store results

	t=0
	p_W_at_t = pi_W   #posterior of W given all information available up to t
	result.append(p_W_at_t)   #just the prior
	
	t=1
	for i in range(0, len(lams_d)):
		p_W_at_t = fuseProbs( p_W_at_t , lams_d[i] )    #fuse in all descriptors
		#p_W_at_t = (p_W_at_t + lams_d[i])/2 
	result.append(p_W_at_t)                        #only store result once
	
	for t in range(2, 2+len(lams_e)):   #for each event time
		p_W_at_t = fuseProbs( p_W_at_t , lams_e[t-2] )    #fuse in all descriptors
		#p_W_at_t = (p_W_at_t + lams_e[t-2])/2 
		result.append(p_W_at_t)                                      #here we store result at each time
	
	print("Result: " + str(result))
	return result

def makeFiltrationPosteriorSequence(result):
	
	res = []    #to store results

	t=0
	p_W_at_t = result[0]   #posterior of W given all information available up to t
	res.append(p_W_at_t)   #just the prior
	
	t=1
	p_W_at_t = fuseProbs( p_W_at_t , result[1] )    #fuse in all descriptors
		#p_W_at_t = (p_W_at_t + lams_d[i])/2 
	res.append(p_W_at_t)                        #only store result once
	
	for t in range(2, 2+len(result)):   #for each event time
		p_W_at_t = fuseProbs( p_W_at_t , result[t-2] )    #fuse in all descriptors
		#p_W_at_t = (p_W_at_t + lams_e[t-2])/2 
		res.append(p_W_at_t)                                      #here we store result at each time
	
	print("Res: " + str(res))
	return res
	

def makeLams_d(seqs, descriptorss, dct_reverse, winMatrix, lDesc, winNum):
	
	#you know freq(d_i | W)  and freq(d_i|L)   -- these are just frequencies 
	freqs_d = makeNotNormalizedFreqs_d(seqs, descriptorss, winMatrix, lDesc)
	
	freqs_d = goodTuring(freqs_d) # + GoodTuring

	#you compute (not normalized)  P(d_i | W)  and P(d_i|L) from the frequencies
	# convert frequencies into probabilities
	lams_d = dict()
	
	for key in freqs_d:
		
		lam_di_given_W = freqs_d[key][0]/(winNum)
		lam_di_given_L = freqs_d[key][1]/(len(seqs) - winNum)
		lams_d[key] = (lam_di_given_W, lam_di_given_L)
		

	#normlize each one:   makeNormalizedLikelihood( p_di_given_W , p_di_given_L):
	for key in lams_d:
		p_di_given_W = makeNormalizedLikelihood(lams_d[key][0] , lams_d[key][1])
		p_di_given_L = makeNormalizedLikelihood(lams_d[key][1] , lams_d[key][0])
		lams_d[key] = (p_di_given_W , p_di_given_L)
	
	print("Lams_d (prob): " + str(lams_d))
	
	#you know which features have occured eg. {3,7,9}
	normalizedLamList_d = [] 
	
	for i in range(0, len(seqs)):
		features = []
		for j in range(0, len(lDesc)):
		
			if any( lDesc[j] in s for s in descriptorss[i]): # single interacting car
				if(j == 10):
					if any( "Distraction_None" in s for s in descriptorss[i]): 
						continue
					else:
						features.append(lams_d[j][0])
				elif(j >= 11):
					features.append(lams_d[11][0])
				else:
					features.append(lams_d[j][0])
			
		normalizedLamList_d.append(features)
	
	return normalizedLamList_d



def makeLams_e(seqs, dct_reverse, winMatrix, winNum):
	
	#you know freq(d_i | W)  and freq(d_i|L)   -- these are just frequencies 
	freqs_e = makeNotNormalizedFreq_e(seqs, dct_reverse, winMatrix)
	
	freqs_e = goodTuring(freqs_e) # + GoodTuring

	#you compute (not normalized)  P(d_i | W)  and P(d_i|L) from the frequencies
	# convert frequencies into probabilities
	lams_e = dict()
	
	for key in freqs_e:
		lam_e_given_W = freqs_e[key][0]/(winNum)
		lam_e_given_L = freqs_e[key][1]/(len(seqs) - winNum)
		lams_e[key] = (lam_e_given_W, lam_e_given_L)

	#normlize each one:   makeNormalizedLikelihood( p_di_given_W , p_di_given_L):
	for key in lams_e:
		p_e_given_W = makeNormalizedLikelihood(lams_e[key][0] , lams_e[key][1])
		p_e_given_L = makeNormalizedLikelihood(lams_e[key][1] , lams_e[key][0])
		lams_e[key] = (p_e_given_W , p_e_given_L)
	
	print("Lams_e (prob): " + str(lams_e))
	
	#you know which features have occured eg. {3,7,9}
	normalizedLamList_e = [] 
	
	for i in range(0, len(seqs)):
		features = []
			
		for label in seqs[i]:
			features.append(lams_e[label][0])
		
		normalizedLamList_e.append(features)
	
	return normalizedLamList_e

# count the number of features in seqs given the winner 
def checkFeatureFreq(feature, seqs, winMatrix):
	win = 0
	lose = 0
	seqnum = []
	for i in range(0, len(seqs)):
		winner = winMatrix[i]
		if feature in seqs[i]:
			if(winner == 1):
				win += 1
			else:
				lose += 1
			seqnum.append(i)
	return win, lose, seqnum

# gaussian process for an interaction
def gpyplot(result, time):

	X = np.vstack(time)
	Y = np.vstack(result)
	#fit and display a Gaussian process
	kernel = GPy.kern.RBF(input_dim=1, ARD=True, variance=1/100, lengthscale=1.) + GPy.kern.White(1)
	m= GPy.models.GPRegression(X,Y,kernel)
	m.optimize(messages=True,max_f_eval = 1000)
	m.plot()
	array = np.arange(len(time))
	array = array[:, None]
	mean_std = m.predict(array)
	print("\n mean_std: " + str(mean_std))
	plt.xlim(0, 20)
	plt.ylim(0,1)
	
	plt.title("GP P(W|D(0:t)) over time")
	plt.xlabel("Time")
	plt.ylabel("P(W|D(0:t))")
	plt.show()
	
	return mean_std

def plotVehicleWins(result, time):
	plt.plot(time, result, linestyle = '-', color='purple')	

	
def plotPedestrianWins(result, time):
	plt.plot(time, result, linestyle = '-', color='lightgreen')

def hist_seq(seqs):
	length = []
	for i in range(0, len(seqs)):
		length.append(len(seqs[i]))
		
	plt.figure()
	plt.hist(length)
	plt.title("Length of sequences")
	plt.xlabel("Length")
	plt.ylabel("Frequency")
	fig = plt.gcf()
	
def hist_time(tDelta):
	plt.figure()
	plt.hist(tDelta)
	plt.title("Duration of interactions")
	plt.xlabel("Time (s)")
	plt.ylabel("Frequency")
	
def getMeanStdforAll(start, end, normalizedLamList_d, normalizedLamList_e):
	
	mean = []
	std = []
	for i in range(start, end):
		#if(len(normalizedLamList_e[i]) == 10):
		result = makeTemporalPosteriorSequence( 74.0/204,  normalizedLamList_d[i],  normalizedLamList_e[i])
		m = gpyplot(result, np.arange(len(normalizedLamList_e[i]) +2))
		print("\n Mean and std : " + str(m))
		mean.append(m[0])	
		std.append(m[1])
		
	#std_average = np.array(std).mean()
	#print("std_average: " + str(std_average))
	#plt.figure()
	#plt.plot(np.arange(len(std)), std_average)

	return mean, std
		


if __name__=="__main__":
	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	lDesc = np.array(["from_Single", "From right", "Overcast", "Sunny", "Raining", "Group", "13-18y", "18-30y", "30-60y", "60+ years", "Distraction", "Female", "female"])
	
	(seqs,descriptorss, dct_reverse, time) = makeSeqs(dir_data, dct_noneEvents)
	
	dct_reverse[62] = 'Ped wins'
	dct_reverse[63] = 'Car wins'

	winMatrix = winnerMatrix(204)	
	
	time = np.reshape(time, (203, 2))
	seqtime, tDelta = duration(time)
	
	
	seqs = makeStopEvent(seqs, winMatrix)  # add stop event to seqs and normalize all sequences' length to 26
	#print(seqs)
	
	normalizedLamList_d = makeLams_d(seqs, descriptorss, dct_reverse, winMatrix, lDesc, 74.0)
	normalizedLamList_e = makeLams_e(seqs, dct_reverse, winMatrix, 74.0)
	
	print("\n Compute Std0: ")
	compute_s_t(normalizedLamList_d, normalizedLamList_e, winMatrix)
	
	
	plt.figure()
	for i in range(0,204):
		result = makeTemporalPosteriorSequence( 74.0/204,  normalizedLamList_d[i],  normalizedLamList_e[i])
		#if(len(normalizedLamList_e[i]) +2 == 10):
		if(winMatrix[i] == 1):
			plotPedestrianWins(result, np.arange(len(normalizedLamList_e[i]) +2))
			#plt.vlines(x=result.index(max(result))-0.75, ymin=0, ymax=max(result) + 0.1, color='black', linestyle=':', zorder=2)
		else:
			#plt.plot(np.arange(len(normalizedLamList_e[i]) +2), result, linestyle='-', color = 'blue', label='V wins')
			plotVehicleWins(result, np.arange(len(normalizedLamList_e[i]) +2))
	
		 
	#gpyplot(result,  np.arange(len(normalizedLamList_e[i]) +2))
		
	plt.xlim(0,30)
	plt.ylim(0,1)

	plt.title("Filtered sequence of P-V interaction with stop events" )#+ str(i))
	plt.xlabel("Time (t)")
	plt.ylabel("P(W|D(0:t))")
	plt.show()

	#hist_seq(seqs)
	
	#hist_time(tDelta)
	
	'''
	means, stds = getMeanStdforAll(149, 150, normalizedLamList_d, normalizedLamList_e)
	print("\n stds: " + str(stds))
	stds_average = [float(sum(col))/len(col) for col in zip(*stds)]
	print("Average std: " + str(stds_average))
	#plt.figure()
	#plt.plot(np.arange(12), stds_average)

	'''
		
	'''
	feature = 27
	win, lose, seqnum = checkFeatureFreq(feature, seqs, winMatrix)
	print("Feature " + str(feature) + " wins: " + str(win))
	print("Feature " + str(feature) + " loses: " + str(lose))
	print("Feature " + str(feature) + " found in seqs: " + str(seqnum))
	'''