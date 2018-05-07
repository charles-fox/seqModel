# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:30:03 2018

@author: fanta
"""

from goTime import  *
from winnerMatrix import *
from entropy import *



def goodTuring(pSeq, pDesc):
	
	for freq in pSeq:	
		if(pSeq[freq][0] == 0):
			pSeq[freq] = (pSeq[freq][0] + 0.001, pSeq[freq][1] - 0.001)
		if(pSeq[freq][1] == 0):
			pSeq[freq] = (pSeq[freq][0] - 0.001, pSeq[freq][1] + 0.001)

	for freq in pDesc:	
		if(pDesc[freq][0] == 0):
			pDesc[freq] = (pDesc[freq][0] + 0.001, pDesc[freq][1] - 0.001)
		if(pDesc[freq][1] == 0):
			pDesc[freq] = (pDesc[freq][0] - 0.001, pDesc[freq][1] + 0.001)

	return pSeq, pDesc


def probDesc(descriptor, winner, pDesc):
	lDesc = np.array(["from_Single", "From right", "Overcast", "Sunny", "Raining", "Group", "13-18y", "18-30y", "30-60y", "60+ years", "Distraction", "Female"])
	
	for j in range(0, len(lDesc)):
		dwin = 0
		dlose = 0
	
		if any( lDesc[j] in s for s in descriptor): # single interacting car
			if(j == 10):
				if any( "Distraction_None" in s for s in descriptor): 
					j += 1
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
def probSeq(seqs, descriptorss, dct_reverse, winMatrix):
	
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
		pDesc = probDesc(descriptorss[i], winner, pDesc)
		
		
	pSeq, pDesc = goodTuring(pSeq, pDesc)
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
		
def initPlot(pDesc):
	print("init")
	
if __name__=="__main__":
	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	
	(seqs,descriptorss, dct_reverse, time) = makeSeqs(dir_data, dct_noneEvents)

	winMatrix = winnerMatrix(204)	
	
	time = np.reshape(time, (203, 2))
	seqtime, tDelta = duration(time)
	
	pSeq, pDesc = probSeq(seqs, descriptorss, dct_reverse, winMatrix)
	
	print(pSeq)
	print("\n")
	print(pDesc)
	
	I_2Dseq = seqInfoGain(pSeq)
	I_2Ddesc = descInfoGain(pDesc)