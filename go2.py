# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:28:35 2018

@author: fanta
"""


#CF 2018-02-12
#code to load the raw CSV questionaaires into python
#makes two structures:
#  seqs[i] gives the sequence of integer-labelled action types of the ith questionaire (there are 205 questionaires in total)
# meta_data[i] stores the "general information" on the ith questionaire. (which might be interesting to cluster the data)
#
# using the data from N drive:  Interact/Observations/P-V/


import os,sys,re
import pdb
import numpy as np
import operator
import findWinner
from fcmotif import findMotif

from sklearn.feature_selection import RFE

from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel


from sklearn import metrics

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report

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


def findWinners(seqs, descriptorss, dct_reverse):
	N = len(seqs)  #now many interactions observed
	winners = np.zeros((N))
	for i in range(0,N):
		seq_human_readable = subseq2human(seqs[i], dct_reverse)
		descriptors_human_readable = descriptorss[i]
		winner = findWinner.findWinner(seq_human_readable, descriptors_human_readable)  
		winners[i]=winner
	return winners
	
	
def addDescription(descriptorss, inputs, seqs, motifs):
	
	for i in range(0, len(descriptorss)):
		if any("from_Single" in s for s in descriptorss[i]): # single interacting car
			inputs[i, 65] = 1		
		if any("From right" in s for s in descriptorss[i]):  # car coming from right
			inputs[i, 66] = 1
		if any("Overcast" in s for s in descriptorss[i]):   # weather overcast
			inputs[i, 67] = 1
		if any("Sunny" in s for s in descriptorss[i]):      # weather sunny
			inputs[i, 68] = 1
		if any("Rainy" in s for s in descriptorss[i]):     # weather rainy
			inputs[i, 69] = 1
		if any("Group" in s for s in descriptorss[i]):     # group of people
			inputs[i, 70] = 1
		if any("(13-18y)" in s for s in descriptorss[i]):  # pedestrian teenager
			inputs[i, 71] = 1
		if any("(18-30y)" in s for s in descriptorss[i]):  # pedestrian young adult
			inputs[i, 72] = 1
		if any("(30-60y)" in s for s in descriptorss[i]):   # pedestrian midage adult
			inputs[i, 73] = 1
		if any("(60+ years)" in s for s in descriptorss[i]):  # pedestrian old adult
			inputs[i, 74] = 1
		if any("Distraction" in s for s in descriptorss[i]):   # pedstrian distracted
			inputs[i, 75] = 1  
		if any("Females" in s for s in descriptorss[i]):     # pedestrian female  # default is 0: male
			inputs[i, 76] = 1
		if any("_Individual female" in s for s in descriptorss[i]):  # pedestrian female # default is 0: male
			inputs[i, 76] = 1
		
	#TODO add presence/absence of motifs to INPUTS
	for i in range(len(seqs)):
		for j in range(len(motifs)):
			for k in range(10):
				inputs[i, 77 + 10*j + k] = findMotif(seqs[i], list(motifs[j][k][0]))
				
	return inputs
	
	
def createDataFrame(dct_reverse, winners):
	
	action = ["Action " + str(x) for x in range(0, len(dct_reverse))]
	desc = np.array(["from_Single", "From right", "Overcast", "Sunny", "Rainy", "Group", "13-18y", "18-30y", "30-60y", "60+ years", "Distraction", "Gender", "Empty1", "Empty2 ", "Empty3", "Empty4"])
	motif2gram =  ["2gram " + str(x) for x in range(1, 11)]
	motif3gram =  ["3gram " + str(x) for x in range(1,11)]
	motif4gram =  ["4gram " + str(x) for x in range(1,11)]
	
	column_names = np.hstack((action, desc))
	column_names = np.hstack((column_names, motif2gram))
	column_names = np.hstack((column_names, motif3gram))
	column_names = np.hstack((column_names, motif4gram))
	column_names = np.hstack((np.array(["Winner"]), column_names))
	
	win = np.reshape(winners, (len(winners), 1))
	dt = np.hstack((win, inputs))	
	dt = np.vstack((column_names, dt))
	
	data = pd.DataFrame(data=dt[1:,0:], index=np.arange(len(winners)), columns=dt[0,0:])
	
	return data

def plotData(data):
	
	print(list(data.columns))
	data.head()
	
	print(data['Winner'].value_counts())
	sns.countplot(x="Winner", data=data, palette='hls')
	plt.show()
	
	pd.crosstab(data['Action 2'], data['Winner']).plot(kind='bar')
	plt.title('Action 2 vs winner')
	plt.xlabel('Action 2')
	plt.ylabel('winner')
	
	pd.crosstab(data['Action 9'], data['Winner']).plot(kind='bar')
	plt.title('Action 9 vs winner')
	plt.xlabel('Action 9')
	plt.ylabel('winner')


	pd.crosstab(data['Action 10'], data['Winner']).plot(kind='bar')
	plt.title('Action 10 vs winner')
	plt.xlabel('Action 10')
	plt.ylabel('winner')

	pd.crosstab(data['Action 11'], data['Winner']).plot(kind='bar')
	plt.title('Action 11 vs winner')
	plt.xlabel('Action 11')
	plt.ylabel('winner')

	pd.crosstab(data['Action 12'], data['Winner']).plot(kind='bar')
	plt.title('Action 12 vs winner')
	plt.xlabel('Action 12')
	plt.ylabel('winner')

	pd.crosstab(data['2gram 1'], data['Winner']).plot(kind='bar')
	plt.title('2 gram 1 vs winner')
	plt.xlabel('2gram 1')
	plt.ylabel('winner')	
	
	pd.crosstab(data['Gender'], data['Winner']).plot(kind='bar')
	plt.title('Pedestrian Gender vs winner')
	plt.xlabel('Pedestrian Gender')
	plt.ylabel('winner')
	

def logitRegression(data):
	
	# Feature Selection
	logistic = LogisticRegression()
	rfe = RFE(logistic, 18)
	rfe = rfe.fit(inputs, winners)
	print(rfe.support_)
	print(rfe.ranking_)
	features = rfe.support_
	print("\nFeature index: " + str(np.where(features == True)))
	
	# creating testing and training set
	X_train,X_test,Y_train,Y_test = train_test_split(inputs,winners,test_size=0.33)
	
	# train scikit learn model 
	clf = LogisticRegression()
	clf.fit(X_train,Y_train)
	score = round(clf.score(X_test,Y_test), 2)
	print('score Scikit learn: ', score)

	logistic.fit(inputs,winners)
	predicted = logistic.predict(X_test)
	print("Predicted: " + str(predicted))
	plt.figure()
	plt.plot(predicted)
	
	# Metrics: confusion matrix
	cm = metrics.confusion_matrix(Y_test, predicted)
	print(cm)
	
	# plot
	plt.figure(figsize=(2,2))
	sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
	plt.ylabel('Actual label');
	plt.xlabel('Predicted label');
	all_sample_title = 'Accuracy Score: {0}'.format(score)
	plt.title(all_sample_title, size = 15)
	plt.show()
	
	# cross validation
	kfold = sklearn.cross_validation.KFold(X_train.shape[0], n_folds=10)
	modelCV = LogisticRegression()
	scoring = 'accuracy'
	results = sklearn.metrics.accuracy_score(Y_test, predicted)
	print("\n\n 10-fold cross validation average accuracy: %.3f" % (results.mean()))
	print("\n")
	
	# precision
	print(classification_report(Y_test, predicted))	
	
	# ROC
	logit_roc_auc = roc_auc_score(Y_test, logistic.predict(X_test))
	fpr, tpr, thresholds = roc_curve(Y_test, logistic.predict_proba(X_test)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Log_ROC')
	plt.show()

	# train with selected features
	train_cols = ['Action 2', 'Action 9', 'Action 10', 'Action 11', 'Action 12', 'Action 13', 'Action 14', 'Action 16', 'Action 18', 'Action 24', 'Action 32', 'Action 41', 'Action 48', 'Action 53', 'Action 57', '2gram 10', '3gram 2', '3gram 9']
	X = data[train_cols]	
	#print(X)
	y = data['Winner']
	logit_model = sm.Logit(y.astype(float), X.astype(float))
	result = logit_model.fit(method='bfgs')
	print(result.summary())
	

if __name__=="__main__":

	dct_noneEvents = loadNoneEvents()

	dir_data = os.environ['ITS_SEQMODEL_DATADIR']
	#eg in ~/.bashrc: export ITS_SEQMODEL_DATADIR=/home/user/data/oscarPedestrians/

	(seqs,descriptorss, dct_reverse) = makeSeqs(dir_data, dct_noneEvents)	
	
	motifs = []

	for ngram in range(2,5):
		print("TOP %i-grams:  (frequency) "%ngram)
		res = rankSubSeqs(seqs, ngram)    #CF simplest possible n-gram finder function 
		motifs.append(res[:10])
		showResults(res, dct_reverse)
		print("")
			

	winners = findWinners(seqs, descriptorss, dct_reverse)  

	#convert presence/absennce of temporal events to features (to use as inputs to machine learning)
	inputs = np.zeros(( len(seqs) , 110 )) # was 100
	for i in range(0, len(seqs)):
		for j in seqs[i]:
			inputs[i, j] = 1
		
	#TODO add presence/absence of descriptors to INPUTS
	inputs = addDescription(descriptorss, inputs, seqs, motifs)

	#TODO do machine learning (eg logit regression) to predict the winners from the input feature matrix
	
	# create correct Pandas data frame format
	data = createDataFrame(dct_reverse, winners)
	
	#plot some statistical information 
	plotData(data)
	
	# logistic regression
	logitRegression(data)
	
	
