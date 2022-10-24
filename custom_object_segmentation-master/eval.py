import argparse

from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *

###############################################
# INPUT IMAGE TO EVALUATE THE MODEL
###############################################
INPUT_IMAGE = './dataset/e_motor/test/grabImg_0_221013_100418.png'
INPUT_IMAGE1 = './dataset/e_motor/test/grabImg_0_221013_100441.png'
INPUT_IMAGE2 = './dataset/e_motor/test/grabImg_0_221013_102352.png'

IMAGE_PATH = INPUT_IMAGE
###############################################
cfg_save_det = "./output/object_detection/OD_cfg.pickle"
cfg_save_seg = "./output/segmentation/S_cfg.pickle"
cfg_save_path = cfg_save_seg

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
    # cfg = get_cfg()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    predictor = DefaultPredictor(cfg)

    output_dir_det = "./output/object_detection"
    output_dir_seg = "./output/segmentation"

    output_dir = output_dir_seg

    img = cv2.imread(IMAGE_PATH)
    if img is not None:
        print('We got input image with shape of ' + str(img.shape))

    else:
        print('There is no input image')

    on_image(IMAGE_PATH, predictor, output_dir)

    # video_path = 'dataset/e_motor/test/'
    # on_video(video_path, predictor)
