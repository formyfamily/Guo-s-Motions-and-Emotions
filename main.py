import os
import pdb
import array
import json
import random
import zipfile
from argparse import ArgumentParser
from utils import ScreenPrinter, DataSet


dataArg = {	
	'movies' : None, # None means all movies
	'visual_features' : {
		"acc" : True,
		"cedd" : True,
		"cl" : True,
		"eh" : True,
		"fc6" : False,
		"fcth" : True,
		"gabor" : True,
		"jcd" : True,
		"lbp" : True,
		"sc" : True,
		"tamura" : True
	},
	'feature_dir' : "MEDIAEVAL17-DevSet-Visual_features",
	'annotation_dir' : "MEDIAEVAL17-DevSet-Valence_Arousal-annotations",
	'uploaded_dir' : "data/uploaded_data"
}

def get_args():
    parser = ArgumentParser(description="Guo's Emotions!")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset = DataSet.DataSet(dataArg)
    pdb.set_trace()
    #ScreenPrinter.getScreenShotData("data/continuous-movies/After_The_Rain.mp4")


if __name__ == "__main__":
    main()