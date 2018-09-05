#!/usr/bin/env python

import h5py
import sys





def applyModel(outputD):
    print('Zeile34')
    from keras.models import load_model  
    print('zeile35') 
    model = load_model("%sMET_model.h5"%outputD)



if __name__ == "__main__":
	
	outputDir = sys.argv[1]
	print(outputDir)
	
	applyModel(outputDir)	
