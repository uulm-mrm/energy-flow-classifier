import os


# nuimg categories
# more classes at https://www.nuscenes.org/nuimages#data-annotation
NUIMG_ID_CATEGORIES = ["human.pedestrian.adult", "vehicle.bicycle", "vehicle.car"]
NUIMG_OOD_CATEGORIES = []

# COCO labels for FasterRCNN
# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# pedestrian is actually person in COCO
COCO_ID_CATEGORIES = {"pedestrian": 1, "bicycle": 2, "car": 3}
COCO_OOD_CATEGORIES = []

# dataset consts
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

NUIM_DATASET_VERSION = "v1.0-val"  # change between train and val for train and test
NUIM_DATASET_ROOT = os.path.join(DATASET_DIR, "nuimages-v1.0")

OOD = False
FEATURES_DATASET_DIR = os.path.join(
    DATASET_DIR, "v1.0-frame_features", "id" if not OOD else "ood", NUIM_DATASET_VERSION
)

# numerical consts
# 0.75 bcs we want really good boxes for features and not mostly good features with 0.5
IOU_THRESH = 0.75
