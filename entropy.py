import numpy as np
import math


freqs = np.array([[36,41],[56,71]])
#Columns=causes(features, eg. genders), Rows=results(outcomes, eg. pedWins,pedLoses)
#eg. there are 36 males who win, 56 males who lose, 41 females who win, 71 females who lose.)

ps = freqs/sum(sum(freqs))

def log2(x):
	return math.log(x)/math.log(2) 

#entropy of 1d distribution, H[X]
def H(ps):
	return -sum(list(map( lambda p: p*log2(p) , ps )))

#joint entropy of a 2d distribution, H[X,Y]
def H_2D(ps):
	(R,C) = ps.shape
	m = np.zeros((R,C))
	for r in range(0,R):
		for c in range(0,C):
			m[r,c] = ps[r,c]*log2( ps[r,c] )
	return -sum(sum(m))

#marginal entropy H[R] of 2D (row,col)
def Hmarginal_2D(ps):
	p_rows = np.sum(ps, axis=1)	
	return H(p_rows)

#conditional entropy H[R|C] of 2D rows(results) and columns(conditions/causes)
def Hcond_2D(ps):
	(R,C) = ps.shape
	m = np.zeros((R,C))
	t = np.zeros((R,C))
	for r in range(0,R):
		for c in range(0,C):
			p_r_and_c   = ps[r,c]
			p_r_given_c = ps[r,c] / ( sum(ps[:, c]))

			t[r,c] = p_r_given_c
			m[r,c] = p_r_and_c*log2( p_r_given_c )
#	print(t)
#	print(m)
	return -sum(sum(m))

def I_2D(ps):
	H_r         = Hmarginal_2D(ps)
	H_r_given_c = Hcond_2D(ps)
	I = H_r - H_r_given_c
	return I 

print("joint distribution:")
print(ps)
print("joint entropy H[r,c]]= %f Sh"% H_2D(ps))
print("marginal entropy H[r]=%f Sh"% Hmarginal_2D(ps))
print("conditional entropy H[r|c]=%f Sh"% Hcond_2D(ps))
print("mutual information I[r,c]=%f Sh"%I_2D(ps) )
