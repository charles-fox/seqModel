# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:38:44 2018

@author: fanta
"""

from Bio.Seq import Seq
from Bio import motifs




"""
convert the sequence number into unique ascii code
"""
def ascii_code(seq):    
    
    offset = 40
    seq_ascii = ''
    for num in seq:
        key_ascii = chr(num + offset) 
        seq_ascii += key_ascii
    return seq_ascii

    
def findMotif(seq, motif):
    
    seq_ascii = ascii_code(seq)
    mtf = ascii_code(motif)
    my_seq = Seq(seq_ascii)
    print(my_seq)
    print(mtf)
    print(my_seq.count(mtf))
    if(my_seq.count(mtf) > 0):
        return 1
    else:
        return 0	
    
    """
    instances = [Seq("ACTG"), Seq("TACA")]
    motifs.alphabet = my_seq.alphabet
    m = motifs.create(instances)
    print(m.counts)
    test_seq=Seq("T?ACTGC'TT/CAACCCAAGCATTA", m.alphabet)
    print(len(test_seq))
    for pos, seq in m.instances.search(test_seq):
        print("%i %s" % (pos, seq))
    """

if __name__=="__main__":
    
    
    seq_ascii = ascii_code(seqs[203])
    print(seqs[203])
    
    findMotif(seqs[203], np.array([5, 74]))