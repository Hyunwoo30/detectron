
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import matplotlib.pyplot as plt


def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1],
                       metadata=dataset_custom_metadata,
                       scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device,
                  output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = train_dataset_name
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg


def on_image(image_path, predictor, output_dir):
    im = cv2.imread(image_path)
    print('Processing the input image... ' )
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    image = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    font_face = cv2.FONT_HERSHEY_DUPLEX
    if len(outputs['instances']) >= 1:
        text = ("Number of WELDED cases detected: " + str(len(outputs['instances'])))
    else:
        text = " WELDED cases have not been detected! "
    org = (30, 40)
    fontScale = 1
    color = (250, 250, 250)
    thickness = 1
    img = image.get_image()
    img = cv2.putText(img, text, org, font_face, fontScale, color, thickness, cv2.LINE_AA)

    filename = './output/outputImage.jpg'
    cv2.imwrite(filename, img)
    print('Successfully saved the Proceeded output image')

    plt.figure(figsize=(15, 20))
    plt.imshow(img)
    plt.show()


def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(
            'Error in opening the file...'
        )
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions['instances'].to('cpu'))

        cv2.imshow('Result', output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        (success, image) = cap.read()
