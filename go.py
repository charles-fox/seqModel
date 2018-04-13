
#CF 2018-02-12
#code to load the raw CSV questionaaires into python
#makes two structures:
#  seqs[i] gives the sequence of integer-labelled action types of the ith questionaire (there are 205 questionaires in total)
# meta_data[i] stores the "general information" on the ith questionaire. (which might be interesting to cluster the data)
#
# using the data from N drive:  Interact/Observations/P-V/


import os,sys,re
import pdb
import operator


def loadNoneEvents():  #load lists of human-selected "non-events" ie those which are not real events such as Null and Unobs
	dct_noneEvents=dict()
	for fn in ["labels_null", "labels_unobs", "labels_desc"]:
		for line in open(fn):
			line=line.strip()
			(ID, label) = re.match("(\d+): '(.*)'", line).groups()
			ID=int(ID)
			dct_noneEvents[label]=ID
	return dct_noneEvents

def showResults(res, dct_reverse):
	i=0
	for r in res:
		(subseq, count) = r
		subseq_human = subseq2human(subseq, dct_reverse)
		print(subseq_human, count)
		i+=1
		if i>10:   #limit to top 10
			break

def subseq2human(subseq, dct_reverse):
	str_out = ""
	for num in subseq:
		label=dct_reverse[num]
		str_out = str_out+label+" , "
	return str_out

def rankSubSeqs(seqs, ngram):
	dct_hyps = dict()     #my best hypotheses about populat sequences. the subseq is the key. the number of times it is seen is stored.
	subSeqLength = ngram   #length of subseqs to look for (EDIT THIS)
	
	for seq in seqs:   #analyse a sequence
		for t in range(0,  len(seq)-subSeqLength):
			subseq = tuple(seq[t:t+subSeqLength])    #tuple is needed for dict hashing

			if subseq in dct_hyps:
				dct_hyps[subseq] = dct_hyps[subseq] + 1
			else:
				dct_hyps[subseq] = 1   #create hypothesis

	res = sorted(dct_hyps.items(), key=operator.itemgetter(1))
	res.reverse()
	return res


def makeSeqs(dir_data, dct_noneEvents=dict()):
	dir_PV = dir_data+"P-V/"
	dct_count=-1
	dct=dict()
	dct_reverse=dict()
	seqs = []   #pronounced "seeks" for sequences
	descriptorss = []    #CF I use sufix "ss" to mean a list of lists. (ie list of many descriptor lists, for all interactions)
	for str_date in os.listdir(dir_PV):
		dir_date = dir_PV+str_date+"/"
		for str_interaction in os.listdir(dir_date):
			if str_interaction[0]=="S":
				continue #Screenshot file	
			if str_interaction[0]=="T":
				continue #Thumbs file	
			fn_interaction = dir_date+str_interaction
			#print(fn_interaction)

			seq = []
			descriptors = []   #list of all descriptors of one interaction.
			for line in open(fn_interaction):
				line=line.strip()
				if len(line)<1:
					continue
				if line[0]=="S":
					continue  #header
				fields=line.split(";")
				if len(fields)>5:
			#		print(fields)
					label = fields[2].strip()+"_"+fields[3].strip()+"_"+fields[4].strip()   #strip fixes a data glitch, sometimes fields have spaces at end, other times not
			#		label = fields[4]
					
					if fields[2]=="General Information":  #these are all descriptors, not events
						descriptors.append(label)
					elif fields[2]=="Graphic":
		#				print("GRAPHIC")
						foo=1

					#test if its a descriptor or an event.  (not all descriptors are "General Information").
					elif label in dct_noneEvents:
						descriptors.append(label)
					else:
						#have we seen this label before?
						if label in dct:
							id_label = dct[label]
						else:
							dct_count+=1

							dct[label]=dct_count				
							id_label = dct[label]
							dct_reverse[dct_count] = label

						seq.append(id_label)
			seqs.append(seq)
			descriptorss.append(descriptors) 
	return (seqs, descriptorss, dct_reverse)

if __name__=="__main__":

	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	#eg in ~/.bashrc: export ITS_SEQMODEL_DATADIR=/home/user/data/oscarPedestrians/

	(seqs,descriptorss, dct_reverse) = makeSeqs(dir_data, dct_noneEvents)	

	for ngram in range(2,5):
		print("TOP %i-grams:  (frequency) "%ngram)
		res = rankSubSeqs(seqs, ngram)    #CF simplest possible n-gram finder function 
		showResults(res, dct_reverse)
		print("")
			
