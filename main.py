import os
import pdb
import array
import json
import random
import zipfile
from argparse import ArgumentParser
from utils import ScreenPrinter

def get_args():
    parser = ArgumentParser(description="Guo's Emotions!")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    ScreenPrinter.getScreenShotData("data/continuous-movies/After_The_Rain.mp4")

if __name__ == "__main__":
    main()