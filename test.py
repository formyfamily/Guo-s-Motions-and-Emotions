import numpy as np 
import os
import pdb
import csv

features = np.array([[1,2,0],
					[3,1,0],
					[4,2,1],
					[5,3,6],
					[2,5,3],
					[1,7,4],
					[3,5,7],
					[1,2,3],
					[6,7,9],
					[2,7,9],
					[2,4,5],
					[8,1,3],
					[4,7,3],
					[1,1,1],
					[0,1,2]])
features = features[0:2*5+5,:].transpose(1, 0)
features = features.reshape(features.shape[0], features.shape[1]//5, 5).sum(2).transpose(1, 0)
pdb.set_trace()
features = features[1:features.shape[0],:]+features[0:features.shape[0]-1,:] 
pdb.set_trace()