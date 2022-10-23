import multiprocessing
import subprocess
import glob
from time import sleep
import os
import random
import tarfile
import shutil
import sys
import csv
import pickle
import numpy as np


path = ""
output = ""

def _crop_list():
    data = glob.glob(path + '*.mp4')
    return sorted(data)

def crop(v):
    if not os.path.exists( output + '%s' % v):
        os.makedirs(output + '%s' % v)
    subprocess.check_call([
    'ffmpeg',
    '-i', '%s' % v,
    '-vf', 'fps=25',
    '-s','299x299',
    output+'/%s' % v.split('/')[-1][:-4] +'/%03d.jpg',
    '-hide_banner'])


def main():
    p = multiprocessing.Pool(32)
    p.map(crop, _crop_list())

if __name__ == "__main__":
    main()
