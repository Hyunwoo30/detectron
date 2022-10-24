
from Detector import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

detector = Detector(model_type="PS")

detector.onImage("images/2.jpg")