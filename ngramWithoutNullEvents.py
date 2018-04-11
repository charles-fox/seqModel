from go import *


if __name__=="__main__":

        dir_data = os.environ['ITS_SEQMODEL_DATADIR']
        #eg. export ITS_SEQMODEL_DATADIR=/home/user/data/oscarPedestrians/

        (seqs,meta_datas, dct_reverse) = makeSeqs(dir_data)
        print(meta_datas)
        print(seqs)                    #use seqs to call your own analysis functions to look for patterns !

        res = rankSubSeqs(seqs)    #CF simplest possible n-gram finder function 
        #print(res)

        showResults(res, dct_reverse)

