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
		dwin = 0
		dlose = 0
	
		if any( lDesc[j] in s for s in descriptor): # single interacting car
			if(j == 10):
				if any( "Distraction_None" in s for s in descriptor): 
					continue
				elif(winner == 1):
					dwin += 1 
				else:
					dlose += 1
			elif(winner == 1):
				dwin += 1 
			else:
				dlose += 1
		
		lam_d[j] = (lam_d[j][0] + dwin, lam_d[j][1] + dlose)  # lam_d[j][0] = lam_d_given_W and lam_d[j][1] = lam_d_given_L
	
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

def makeLams_d(seqs, descriptorss, dct_reverse, winMatrix, lDesc, winNum):
	
	#you know freq(d_i | W)  and freq(d_i|L)   -- these are just frequencies 
	freqs_d = makeNotNormalizedFreqs_d(seqs, descriptorss, winMatrix, lDesc)
	
	freqs_d = goodTuring(freqs_d) # + GoodTuring

	#you compute (not normalized)  P(d_i | W)  and P(d_i|L) from the frequencies
	# convert frequencies into probabilities
	lams_d = dict()
	#for key in freqs_d:    
	#    lams_d[key] = tuple(t/(len(seqs)+2) for t in freqs_d[key])
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
				else:
					features.append(lams_d[j][0])
			
		normalizedLamList_d.append(features)
		
	#return  normalizedLamList[ 3,7,9] 
	#print("NormalizedLamList_d: " + str(normalizedLamList_d))
	
	return normalizedLamList_d



def makeLams_e(seqs, dct_reverse, winMatrix, winNum):
	
	#you know freq(d_i | W)  and freq(d_i|L)   -- these are just frequencies 
	freqs_e = makeNotNormalizedFreq_e(seqs, dct_reverse, winMatrix)
	
	freqs_e = goodTuring(freqs_e) # + GoodTuring

	#you compute (not normalized)  P(d_i | W)  and P(d_i|L) from the frequencies
	# convert frequencies into probabilities
	lams_e = dict()
	#for key in freqs_e:    
	 #   lams_e[key] = tuple(t/(len(seqs)+2) for t in freqs_e[key])
	
	for key in freqs_e:
		lam_e_given_W = freqs_e[key][0]/(winNum)
		lam_e_given_L = freqs_e[key][1]/(len(seqs) - winNum)
		lams_e[key] = (lam_e_given_W, lam_e_given_L)

	#normlize each one:   makeNormalizedLikelihood( p_di_given_W , p_di_given_L):
	for key in lams_e:
		if(key == 48):
			print("Hello\n") 
		p_e_given_W = makeNormalizedLikelihood(lams_e[key][0] , lams_e[key][1])
		p_e_given_L = makeNormalizedLikelihood(lams_e[key][1] , lams_e[key][0])
		lams_e[key] = (p_e_given_W , p_e_given_L)
	
	#print("Lams_e (prob): " + str(lams_e))
	
	#you know which features have occured eg. {3,7,9}
	normalizedLamList_e = [] 
	
	for i in range(0, len(seqs)):
		features = []
			
		for label in seqs[i]:
			features.append(lams_e[label][0])
		
		normalizedLamList_e.append(features)
		
	#return  normalizedLamList[ 3,7,9] 
	#print("NormalizedLamList_e: " + str(normalizedLamList_e))
	
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

#compute the average of filtered sequences
#def average(results):
	
def plotResult(result, time):
	
	x_obs = time
	true_model = result	
	
	noise = np.random.normal(0, 1, x_obs.shape)
	y_obs = np.polyval(true_model, x_obs) + noise
	
	# Fit to a 5-th order polynomial
	fit_model = np.polyfit(x_obs, y_obs, 11)
	
	x = np.linspace(0, 30, 10)
	y_true = np.polyval(true_model, x)
	y_pred = np.polyval(fit_model, x)
	
	# Made up confidence intervals (I'm too lazy to do the math...)
	high_bound = y_pred + 0.1 * (0.1 * x**4 + 0.1)
	low_bound = y_pred - 0.1 * (0.1 * x**4 + 0.1)
	
	# Plot the results...
	fig, ax = plt.subplots()
	ax.fill_between(x, high_bound, low_bound, color='gray', alpha=0.5)
	ax.plot(x_obs, y_obs, 'ko', label='Observed Values')
	ax.plot(x, y_pred, 'k--', label='Predicted Model')
	ax.plot(x, y_true, 'r-', label='True Model')
	ax.legend(loc='upper left')
	plt.show()

def gpyplot(result, time):
	
	
	#generate some test data
	X = np.vstack(time)
	Y = np.vstack(result)
	#fit and display a Gaussian process
	kernel = GPy.kern.RBF(input_dim=1, ARD=True, variance=1/100, lengthscale=1.) + GPy.kern.White(1)
	m= GPy.models.GPRegression(X,Y,kernel)
	m.optimize(messages=True,max_f_eval = 1000)
	m.plot()
	plt.xlim(0, 20)
	plt.ylim(0,1)
	
	plt.title("P(W|D(0:t)) over time")
	plt.xlabel("Time")
	plt.ylabel("P(W|D(0:t))")
	plt.show()

def addEndGameVehicle(result, time, seq):
	endgame = False
	for i in range(0, len(seq)):
		if(endgame == False and (seq[i] == 9 or seq[i] == 13 or seq[i] == 21 or seq[i] == 25 or seq[i] == 19 or seq[i] == 29 or seq[i]== 32 or seq[i] == 38 or seq[i] == 24)):
			plt.plot(time[i+2], result[i+2], marker='o', color='cyan')
			plt.vlines(x=time[i+2], ymin=result[i+2]-0.1, ymax= result[i+2]+0.1, color='blue', zorder=2)
			plt.plot(np.arange(i+3), result[:i+3], linestyle='-', color = 'green', label='V wins')
			plt.plot(np.arange(i+2,len(time)) , result[i+2:], linestyle=':', color = 'green', label='V wins')
			endgame = True
		else:
			continue
	
	
def addEndGamePedestrian(result, time, seq):
	endgame = False
	for i in range(0, len(seq)):
		if(endgame == False):
			if(seq[i] == 27 or seq[i] == 31 or seq[i] == 41 or seq[i] == 48 or seq[i] == 61 or seq[i] == 45 or seq[i] == 26 or result[i] > 0.6):
				plt.plot(time[i+2], result[i+2], marker='o', color='red')
				plt.vlines(x=time[i+2], ymin=result[i+2]-0.1, ymax= result[i+2]+0.1, color='purple', zorder=2)
				plt.plot(np.arange(i+3), result[:i+3], linestyle='-', color = 'red', label='P wins')
				#plt.plot(np.arange(i+2,len(time)) , result[i+2:], linestyle=':', color = 'red', label='P wins')
				#plt.plot([i+2, i+5], [result[i+2], 0.5], marker='o')
				plt.plot(np.arange(i+2,len(time)) , result[i+2:] , linestyle=':', color = 'red', label='P wins')
				endgame = True
		else:
			continue

def histogram(seqs):
	length = []
	for i in range(0, len(seqs)):
		length.append(len(seqs[i]))
		
	plt.figure()
	plt.hist(length)
	plt.title("Sequence Length")
	plt.xlabel("Length")
	plt.ylabel("Frequency")
	fig = plt.gcf()


if __name__=="__main__":
	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	lDesc = np.array(["from_Single", "From right", "Overcast", "Sunny", "Raining", "Group", "13-18y", "18-30y", "30-60y", "60+ years", "Distraction", "Female"])
	
	(seqs,descriptorss, dct_reverse, time) = makeSeqs(dir_data, dct_noneEvents)

	winMatrix = winnerMatrix(204)	
	
	time = np.reshape(time, (203, 2))
	seqtime, tDelta = duration(time)
		
	normalizedLamList_d = makeLams_d(seqs, descriptorss, dct_reverse, winMatrix, lDesc, 74.0)
	normalizedLamList_e = makeLams_e(seqs, dct_reverse, winMatrix, 74.0)
	
	plt.figure()
	for i in range(149, 150):
		result = makeTemporalPosteriorSequence( 74.0/204,  normalizedLamList_d[i],  normalizedLamList_e[i])
		#plt.axvline(x=result.index(max(result)), color='k', linestyle='--')
		if(len(normalizedLamList_e[i]) +2 >= 2):
			if(winMatrix[i] == 1):
				addEndGamePedestrian(result, np.arange(len(normalizedLamList_e[i]) +2), seqs[i])
				#plt.vlines(x=result.index(max(result))-0.75, ymin=0, ymax=max(result) + 0.1, color='black', linestyle=':', zorder=2)
				
			else:
				plt.plot(np.arange(len(normalizedLamList_e[i]) +2), result, linestyle='-', color = 'blue', label='V wins')
				#addEndGameVehicle(result, np.arange(len(normalizedLamList_e[i]) +2), seqs[i])
				
				 
		
		#plotResult(result, np.arange(len(normalizedLamList_e[i]) +2))
		#gpyplot(result, np.arange(len(normalizedLamList_e[i]) +2))		
		plt.xlim(0,30)
		plt.ylim(0,1)
	
	plt.title("Filtered sequence of P-V interaction" )#+ str(i))
	plt.xlabel("Time (t)")
	plt.ylabel("P(W|D(0:t))")
	plt.show()
	#plt.gca().legend(('P wins','V wins'))
		
	histogram(seqs)
	
	'''	
	feature = 48
	win, lose, seqnum = checkFeatureFreq(feature, seqs, winMatrix)
	print("Feature " + str(feature) + " wins: " + str(win))
	print("Feature " + str(feature) + " loses: " + str(lose))
	print("Feature " + str(feature) + " found in seqs: " + str(seqnum))
	'''