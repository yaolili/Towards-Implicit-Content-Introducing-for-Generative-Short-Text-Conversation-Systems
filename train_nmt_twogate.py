import numpy
import os

from nmt_all_twogate import train

def main(job_id, params):
    print(params)
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=80,
                     valid_batch_size=80,
                     validFreq=5000,
                     dispFreq=500,
                     saveFreq=5000,
                     sampleFreq=5000,
                     datasets=['../data/train.query',
                     '../data/train.reply',
                     '../data/train.keywords'],
                     valid_datasets=[
                     '../data/dev.query',
                     '../data/dev.reply',
                     '../data/dev.keywords'],
                     dictionaries=[
                     '../data/dict.pkl',
                     '../data/dict.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['../model/model.npz'],
        'dim_word': [610],
        'dim': [1000],
        'n-words': [63000],  
        'optimizer': ['adadelta'], 
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})

