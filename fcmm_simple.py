from go import *
import numpy as np

""" Create probability matrix with sequence frequencies"""
def prob_matrix():
    
    (seqs, meta_datas, dct_reverse, file_counter) = makeSeqs(dir_data)	
    res = rankSubSeqs(seqs)    #CF simplest possible n-gram finder function 
    
    Tfreq = np.zeros((181, 181))
    for pair in res:
        Tfreq[pair[0][0], pair[0][1]] = pair[1]
    print(Tfreq)
    print("save csv\n")
    np.savetxt('Tfreq.csv', Tfreq, delimiter=',')
    
    # normalize the matrix to obtain probabilities summing up to 1
    norm_vec = Tfreq[:,:].sum(axis=1)
    print(norm_vec)
    Tnorm = Tfreq/ norm_vec[:,None]
    print(Tnorm)
    print("save csv\n")
    np.savetxt('Tnorm.csv', Tnorm, delimiter=',')
  
    return Tnorm, res, dct_reverse

"""
# create new list of ngrams with probalilities  
# result contains the ngrams with frequencies
# T is the normalized transisition matrix with probabilities
"""
def add_prob(result, T):    
    res_prob = []    
    for val in result:
        print(val)
        res_prob.append(((val[0][0], val[0][1]), T[val[0][0], val[0][1]]))
    print(res_prob)
    return res_prob 


""" initial state of possible actions (might need some changes) """
def initial_state():
    init_state = np.ones(181)/181  
    print(init_state)
    return init_state


"""
# compute p(A, B | C) = p(A | C) * p(B | C)
# Tm is thte transition matrix 
"""
def markov_model(a, b, c, Tm):
    print("Tm[a,c] = " + str(Tm[a,c]))   # P(a|c) 
    print("Tm[b,c] = " + str(Tm[b,c]))   # P(b|c)
    prob = Tm[a,c]*Tm[b,c]
    return prob
   
   
""" Same as markov model but adding some noise to the data """
def hmm_model(a,b,c, Tm, noise):
    print("Tm[a,c] = " + str(Tm[a,c]))   # P(a|c) 
    print("Tm[b,c] = " + str(Tm[b,c]))   # P(b|c)
    prob = (0.9)*Tm[a,c]*Tm[b,c] + 0.1*noise
    return prob
    
    
if __name__=="__main__":
    
    T, res, dct_reverse = prob_matrix()
    Result = add_prob(res, T)
    #showResults(Result, dct_reverse)
    
    print("\nMarkov Model version")
    # example of possible sequence
    pmm = markov_model(12, 11, 13, T)
    #print("\n")
    print("PROB OF " + str(dct_reverse[12]) + " AND " + str(dct_reverse[11]) + " GIVEN " + str(dct_reverse[13]) + " IS  " + str(pmm))

    print("\nHMM version")
    # example of possible sequence
    phmm = hmm_model(12, 11, 13, T, 0.1)
    print("PROB OF " + str(dct_reverse[12]) + " AND " + str(dct_reverse[11]) + " GIVEN " + str(dct_reverse[13]) + " IS  " + str(phmm))
