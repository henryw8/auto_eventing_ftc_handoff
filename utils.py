import bz2
import pickle
import _pickle as cPickle

# Save/load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)