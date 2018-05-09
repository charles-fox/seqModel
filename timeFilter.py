# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:30:03 2018

@author: fanta
"""

from goTime import  *
from winnerMatrix import *
from entropy import *
import matplotlib.pyplot as plt


# to deal with zeros for log funcion
def goodTuring(pSeq, pDesc):
	
	for freq in pSeq:	
		#if(pSeq[freq][0] == 0):
		pSeq[freq] = (pSeq[freq][0] + 1, pSeq[freq][1] + 1)
		#if(pSeq[freq][1] == 0):
		#pSeq[freq] = (pSeq[freq][0] + 1, pSeq[freq][1] + 1)

	for freq in pDesc:	
		#if(pDesc[freq][0] == 0):
		pDesc[freq] = (pDesc[freq][0] + 1, pDesc[freq][1] + 1)
		#if(pDesc[freq][1] == 0):
		#pDesc[freq] = (pDesc[freq][0] + 1, pDesc[freq][1] + 1)

	return pSeq, pDesc


def probDesc(lDesc, descriptor, winner, pDesc):
	
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
		
		pDesc[j] = (pDesc[j][0] + dwin, pDesc[j][1] + dlose)
	
	return pDesc
						
# compute prob of eavh sequence and descriptors given matrix of winner
def probSeq(seqs, descriptorss, dct_reverse, winMatrix, lDesc):
	
	pSeq = dict()
	pDesc = dict()
	
	for i in range(12):
		pDesc[i] = (0,0)
		
	for i in range(0, len(dct_reverse)):
		pSeq[i] = (0,0)
	
	for i in range(0, len(seqs)):
		#seq_human_readable = subseq2human(seqs[i], dct_reverse)
		winner = winMatrix[i]
		#descriptors_human_readable = descriptorss[
		
		for label in dct_reverse:
			pwin = 0
			plose = 0
			#if(seq_human_readable.find(dct_reverse[label]) > 0):
			if(label in seqs[i]):
				if(winner == 1):
					pwin += 1	
				else:
					plose += 1					
			pSeq[label] = (pSeq[label][0] + pwin, pSeq[label][1] + plose)
		pDesc = probDesc(lDesc, descriptorss[i], winner, pDesc)
		
	pSeq, pDesc = goodTuring(pSeq, pDesc)
	
	#print(pDesc)
	
	# convert frequencies into probabilities
	for key in pSeq:    
	    pSeq[key] = tuple(t/(len(winMatrix)+2) for t in pSeq[key])
					
	for key in pDesc:    
	     pDesc[key] = tuple(t/(len(winMatrix)+2) for t in pDesc[key])
						
	#print(pDesc)

	return pSeq, pDesc
	
	
		
# compute Information gain for all sequences	
def seqInfoGain(pSeq):

	I_2Dseq = []	
	
	print("\nSequences:\n ")
	for freq in pSeq:
		freqs = np.array([[pSeq[freq][0], 77 - pSeq[freq][0]], [pSeq[freq][1], 127 - pSeq[freq][1]]])
		print(freqs)		
		
		ps = freqs/sum(sum(freqs))		
		print("marginal entropy H[r]=%f Sh"% Hmarginal_2D(ps))
		print("mutual information I[r,c]=%f Sh"%I_2D(ps) )
		I_2Dseq.append(I_2D(ps))
		
	return I_2Dseq
	
	
# compute Information gain for all descriptors 	
def descInfoGain(pDesc):

	I_2Ddesc = []	
	
	print("\nDescriptors:\n ")
	for freq in pDesc:
		freqs = np.array([[pDesc[freq][0], 77 - pDesc[freq][0]], [pDesc[freq][1], 127 - pDesc[freq][1]]])
		print(freqs)		
		
		ps = freqs/sum(sum(freqs))		
		print("marginal entropy H[r]=%f Sh"% Hmarginal_2D(ps))
		print("mutual information I[r,c]=%f Sh"%I_2D(ps) )
		I_2Ddesc.append(I_2D(ps))
		
	return I_2Ddesc
	
	
# compute p(W|F_i) = p(F_i|W)*p(W)/(p(F_i|W)*p(W) + p(F_i|L)*p(L)) 
def pbSeqDesc(pSeq,pDesc, pi_W):
	
	pbSeq = []
	pbDesc_W = []
	pbDesc_L = []
	
	pi_L = 1 - pi_W
		
	for freq in pDesc:
		pbDesc_W.append(pDesc[freq][0]*pi_W / (pDesc[freq][0]*pi_W + pDesc[freq][1]*pi_L))
		pbDesc_L.append(pDesc[freq][1]*pi_L / (pDesc[freq][1]*pi_L + pDesc[freq][0]*pi_W))
	
	p_d_given_W = pi_W*np.array(pbDesc_W).prod()	
	p_d_given_L = pi_L*np.array(pbDesc_L).prod()	
	
	p_W_given_d = p_d_given_W/(p_d_given_W + p_d_given_L)
	print("\n P_W_given_d: " + str(p_W_given_d))
	
	

	#for freq in pSeq:
	#	pbSeq_W.append(pSeq[freq][0] / (pSeq[freq][0] + pSeq[freq][1]*(1-pWin)))
	#	pbSeq_L.append(pDesc[freq][1]*(1-pWin) / (pDesc[freq][1]*(1-pWin) + pDesc[freq][0]*(pWin)))	
	
	return pbSeq, pbDesc_W, pbDesc_L, p_W_given_d
	
	
	
	
	
# compute cumulative probabilities for each descriptor
# prod_i of p(w|d_i)	
#def initProb(I_2pDdesc, lDesc, descriptorss):
def initProb(pbDesc, lDesc, descriptorss):
	print("\n init \n")
	
	initprob = []
	for descriptor in descriptorss:
		for i in range(0, len(lDesc)):
			if any( lDesc[i] in s for s in descriptor): # single interacting car
				if(i == 0):
					#p = I_2Ddesc[i]	
					p = pbDesc[i]
				elif(i == 10):
					if any( "Distraction_None" in s for s in descriptor): 
						continue
					else:
						#p *= I_2Ddesc[i]	
						p *= pbDesc[i]
				else:
					#p *= I_2Ddesc[i]
					p *= pbDesc[i]
				
		initprob.append(p)
	print(initprob)
	return initprob
	

# compute cumulative probabilities for each sequence
#def seqProb(I_2Dseq, seqs):
def seqProb(pbSeq, seqs):	
	print("\n seqProb \n")	
	
	pseq = []
		
	for i in range(0, 10):
		print("i :" + str(i))
		p = []
		for j in range(0, len(seqs[i])):
			if(len(seqs[i]) > 0):
				#print("j: " + str(j))
				if(j == 0):
					#p.append(I_2Dseq[seqs[i][j]])
					p.append(pbSeq[seqs[i][j]])
					#print(pbSeq[seqs[i][j]])
				else:
					#print(I_2Dseq[seqs[i][j-1]])
					#print(I_2Dseq[seqs[i][j]] )
					#print(pbSeq[seqs[i][j-1]])
					#print(pbSeq[seqs[i][j]])
					#p.append(I_2Dseq[seqs[i][j-1]] * I_2Dseq[seqs[i][j]])
					p.append(pbSeq[seqs[i][j-1]] * pbSeq[seqs[i][j]])
			
		#print(p)	
		pseq.append(p)
	
	print(pseq)
	return pseq
	
# compute X (time) and Y(prob) for plot 	
def prepapreXY(np_d_given_W, np_fit_given_W, timepb, pi_W): # np = normalized likelihood
	
	print("\n prepareXY \n")	
	
	
	if(len(seqpb) > 0):
		x = 	[]
		x.append(0)
		x.append(timepb / len(seqpb)) 
		
		y = []
		y.append(pi_W)
		y.append(np_d_given_W)
	
		for i in range(0, len(seqpb)):
			x.append(timepb / len(seqpb)) 
		
		for i in range(0 , len(seqpb)):
			y.append(pi_W*np_fit_given_W[i] *np_d_given_W / (pi_W*np_fit_given_W[i] *np_d_given_W) + pi_W* (1 - np_fit_given_W[i]) * (1-np_d_given_W))
			
		x = np.cumsum(x)	
		y = np.cumprod(y)
	
		#print("X : " + str(x))
		#print("Y : " + str(y))
		
		return x, y
	else:
		return 0,0
	
	
def plotXY(X, Y):
	
	#plt.plot(X, np.log(Y))
	plt.plot(X, Y)
	plt.title("P(W|Data(i:t)) in function of the time")
	plt.xlabel("Time (s)")
	plt.ylabel("P(W|Data(i:t))")
	plt.show()
	
	
if __name__=="__main__":
	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	lDesc = np.array(["from_Single", "From right", "Overcast", "Sunny", "Raining", "Group", "13-18y", "18-30y", "30-60y", "60+ years", "Distraction", "Female"])
	
	(seqs,descriptorss, dct_reverse, time) = makeSeqs(dir_data, dct_noneEvents)

	winMatrix = winnerMatrix(204)	
	
	time = np.reshape(time, (203, 2))
	seqtime, tDelta = duration(time)
	
	pSeq, pDescW = probSeq(seqs, descriptorss, dct_reverse, winMatrix, lDesc)
	
	print(pSeq)
	print("\n")
	print(pDesc)
	
	'''
	I_2Dseq = seqInfoGain(pSeq)
	I_2Ddesc = descInfoGain(pDesc)
	
	initprob = initProb(I_2Ddesc, lDesc, descriptorss)
	
	initpseq = seqProb(I_2Dseq, seqs)
	
	plt.figure()
	for i in range(0, 10):
		X, Y = prepapreXY(initprob[i], initpseq[i], tDelta[i])
		plotXY(X, Y)
		
	
	'''
	
	
	pbSeq, pbDesc_W, pbDecs_L, p_W_given_d = pbSeqDesc(pSeq,pDesc, 0.38)

	
	initprob = initProb(pbDesc, lDesc, descriptorss)
	
	initpseq = seqProb(pbSeq, seqs)
	
	plt.figure()
	for i in range(0, 10):
		X, Y = prepapreXY(initprob[i], initpseq[i], tDelta[i], 0.38)
		plotXY(X, Y)
		
	
	