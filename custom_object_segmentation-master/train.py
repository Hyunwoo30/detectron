from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import os
import pickle
from utils import *



#######################################################
#  DATA PATH FOR TRAINING
#######################################################

train_dataset_name = "e_motor_train"
train_image_path = "dataset/e_motor/train"
train_json_annot_path = "dataset/e_motor/train/e_motor_train.json"

val_dataset_name = "e_motor_val"
val_image_path = "dataset/e_motor/val"
val_json_annot_path = "dataset/e_motor/valid/e_motor_val.json"
#######################################################


# -----------------------------------------------
config_det = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
checkpoint_det = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

config_seg = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
checkpoint_seg = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"

config_file_path = config_seg
checkpoint_url = checkpoint_seg

# --------------------------------------------
output_dir_det = "./output/object_detection"
output_dir_seg = "./output/segmentation"

output_dir = output_dir_seg
# ------------------------------------------
num_classes = 2

device = "cpu"


# ---------------------------------------
cfg_save_det = "./output/object_detection/OD_cfg.pickle"
cfg_save_seg = "./output/segmentation/S_cfg.pickle"

cfg_save_path = cfg_save_seg
# ----------------------------------------
register_coco_instances(name=train_dataset_name, metadata={},
                        json_file=train_json_annot_path, image_root=train_image_path)

register_coco_instances(name=val_dataset_name, metadata={},
                        json_file=val_json_annot_path, image_root=val_image_path)


# plot_samples(dataset_name=train_dataset_name, n=2) # plot random image for test


def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device,
                        output_dir)

    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()


if __name__ == '__main__':
    main()
