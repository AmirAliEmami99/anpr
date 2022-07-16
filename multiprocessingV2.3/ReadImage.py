import json
import os
import subprocess
import ConFindercopy
import cv2
import Partioningcopy
import numpy as np
import tensorflow as tf
import time

def ReadImage1(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
