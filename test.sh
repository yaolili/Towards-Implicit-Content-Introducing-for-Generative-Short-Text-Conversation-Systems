#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=cpu,floatX=float32

python translate.py -n -p 10  
	../model/model.npz  \
	../data/dict.pkl \
	../data/dict.pkl \
    ../data/test.query \
    ../data/test.topic \
    ./test.predict.reply
    
 
