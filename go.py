
#CF 2018-02-12
#code to load the raw CSV questionaaires into python
#makes two structures:
#  seqs[i] gives the sequence of integer-labelled action types of the ith questionaire (there are 205 questionaires in total)
# meta_data[i] stores the "general information" on the ith questionaire. (which might be interesting to cluster the data)
#
# using the data from N drive:  Interact/Observations/P-V/


import os
import operator

def showResults(res, dct_reverse):
	for r in res:
		(subseq, count) = r
		subseq_human = subseq2human(subseq, dct_reverse)
		print(subseq_human, count)

def subseq2human(subseq, dct_reverse):
	str_out = ""
	for num in subseq:
		label=dct_reverse[num]
		str_out = str_out+label+" , "
	return str_out

def rankSubSeqs(seqs):
	dct_hyps = dict()     #my best hypotheses about populat sequences. the subseq is the key. the number of times it is seen is stored.
	for subSeqLength in range(4,5):   #length of subseqs to look for
		
		for seq in seqs:   #analyse a sequence
			for t in range(0,  len(seq)-subSeqLength):
				subseq = tuple(seq[t:t+subSeqLength])    #tuple is needed for dict hashing

				if subseq in dct_hyps:
					dct_hyps[subseq] = dct_hyps[subseq] + 1
				else:
					dct_hyps[subseq] = 1

	res = sorted(dct_hyps.items(), key=operator.itemgetter(1))
	return res


def makeSeqs(dir_data):
	dir_PV = dir_data+"P-V/"
	dct_count=-1
	dct=dict()
	dct_reverse=dict()
	seqs = []   #pronounced "seeks" for sequences
	meta_datas = []
	for str_date in os.listdir(dir_PV):
		dir_date = dir_PV+str_date+"/"
		for str_interaction in os.listdir(dir_date):
			if str_interaction[0]=="S":
				continue #Screenshot file	
			if str_interaction[0]=="T":
				continue #Thumbs file	
			fn_interaction = dir_date+str_interaction
			print(fn_interaction)

			seq = []
			meta_data = []
			for line in open(fn_interaction):
				line=line.strip()
				if len(line)<1:
					continue
				if line[0]=="S":
					continue  #header
				fields=line.split(";")
				if len(fields)>5:
			#		print(fields)
					label = fields[2]+"_"+fields[3]+"_"+fields[4]
			#		label = fields[4]
					
					if fields[2]=="General Information":
		#				print("GI")
						meta_data.append(label)
					elif fields[2]=="Graphic":
		#				print("GRAPHIC")
						foo=1
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
			meta_datas.append(meta_data)
	return (seqs, meta_datas, dct_reverse)

if __name__=="__main__":

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	#eg. export ITS_SEQMODEL_DATADIR=/home/user/data/oscarPedestrians/

	(seqs,meta_datas, dct_reverse) = makeSeqs(dir_data)	
	print(meta_datas)
	print(seqs)                    #use seqs to call your own analysis functions to look for patterns !

	res = rankSubSeqs(seqs)    #CF simplest possible n-gram finder function 
	#print(res)
	
	showResults(res, dct_reverse)
			
