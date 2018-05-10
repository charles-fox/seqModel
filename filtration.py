# -*- coding: utf-8 -*-
"""
Created on Wed May  9 18:04:58 2018

@author: fanta
"""

from goTime import  *
from winnerMatrix import *
from entropy import *
import matplotlib.pyplot as plt



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
def makeNotNormalizedLams_d(seqs, descriptor, winMatrix, lDesc):

	lams_d = dict()
	
	for i in range(12):
		lams_d[i] = (0,0)
	
	for i in range(0, len(seqs)):
		winner = winMatrix[i]
		lams_d = makeFreqLams_d(lDesc, descriptorss[i], winner, lams_d)  # lam_d: list of likelihood for descriptors as (p(d_i|w), p(d_i|L))
				
	print("lam_d: " + str(lams_d))
						
	return lams_d   
	

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
	
	print("lams_e: " + str(freqs_e))
						
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
	result.append(p_W_at_t)                        #only store result once
	
	for t in range(2, 2+len(lams_e)):   #for each event time
		#p_W_at_t = fuseProbs( p_W_at_t , lams_e[t-2] )    #fuse in all descriptors
		p_W_at_t = (p_W_at_t + lams_e[t-2])/2 
		result.append(p_W_at_t)                                      #here we store result at each time
	
	print("Result: " + str(result))
	return result

def makeLams_d(seqs, descriptorss, dct_reverse, winMatrix, lDesc):
	
	#you know freq(d_i | W)  and freq(d_i|L)   -- these are just frequencies 
	freqs_d = makeNotNormalizedLams_d(seqs, descriptorss, winMatrix, lDesc)
	
	freqs_d = goodTuring(freqs_d) # + GoodTuring

	#you compute (not normalized)  P(d_i | W)  and P(d_i|L) from the frequencies
	# convert frequencies into probabilities
	lams_d = dict()
	for key in freqs_d:    
	    lams_d[key] = tuple(t/(len(seqs)+2) for t in freqs_d[key])

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
	print("NormalizedLamList_d: " + str(normalizedLamList_d))
	
	return normalizedLamList_d



def makeLams_e(seqs, dct_reverse, winMatrix):
	
	#you know freq(d_i | W)  and freq(d_i|L)   -- these are just frequencies 
	freqs_e = makeNotNormalizedFreq_e(seqs, dct_reverse, winMatrix)
	
	freqs_e = goodTuring(freqs_e) # + GoodTuring

	#you compute (not normalized)  P(d_i | W)  and P(d_i|L) from the frequencies
	# convert frequencies into probabilities
	lams_e = dict()
	for key in freqs_e:    
	    lams_e[key] = tuple(t/(len(seqs)+2) for t in freqs_e[key])

	#normlize each one:   makeNormalizedLikelihood( p_di_given_W , p_di_given_L):
	for key in lams_e:
		if(key == 48):
			print("Hello\n") 
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
		
	#return  normalizedLamList[ 3,7,9] 
	print("NormalizedLamList_e: " + str(normalizedLamList_e))
	
	return normalizedLamList_e

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

if __name__=="__main__":
	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	lDesc = np.array(["from_Single", "From right", "Overcast", "Sunny", "Raining", "Group", "13-18y", "18-30y", "30-60y", "60+ years", "Distraction", "Female"])
	
	(seqs,descriptorss, dct_reverse, time) = makeSeqs(dir_data, dct_noneEvents)

	winMatrix = winnerMatrix(204)	
	
	time = np.reshape(time, (203, 2))
	seqtime, tDelta = duration(time)
		
	normalizedLamList_d = makeLams_d(seqs, descriptorss, dct_reverse, winMatrix, lDesc)
	normalizedLamList_e = makeLams_e(seqs, dct_reverse, winMatrix)
	
	plt.figure()
	for i in range(148,153):
		result = makeTemporalPosteriorSequence( 74.0/204,  normalizedLamList_d[i],  normalizedLamList_e[i])
		plt.plot(np.arange(len(normalizedLamList_e[i]) +2), result)
		plt.xlim(0,20)
		plt.ylim(0,1)
	plt.title("Filtered sequence of P-V interaction " + str(i))
	plt.xlabel("Time (s)")
	plt.ylabel("P(W|D(0:t)")
	'''	
	feature = 51
	win, lose, seqnum = checkFeatureFreq(feature, seqs, winMatrix)
	print("Feature " + str(feature) + " wins: " + str(win))
	print("Feature " + str(feature) + " loses: " + str(lose))
	print("Feature " + str(feature) + " found in seqs: " + str(seqnum))
	'''