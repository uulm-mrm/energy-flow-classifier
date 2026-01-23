import os


# nuimg categories
# more classes at https://www.nuscenes.org/nuimages#data-annotation
NUIMG_ID_CATEGORIES = [
    "human.pedestrian.adult",
    "vehicle.bicycle",
    "vehicle.car",
]
COCO_OOD_CATEGORIES = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "truck": 8,
}

# dataset consts
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

NUIM_DATASET_VERSION = "v1.0-val"  # change between train and val for train and test
NUIM_DATASET_ROOT = os.path.join(DATASET_DIR, "nuimages-v1.0")

COCO_DATASET_VERSION = "val2017"
COCO_DATASET_ROOT = os.path.join(DATASET_DIR, "COCO", COCO_DATASET_VERSION)

OOD = True
FEATURES_DATASET_DIR = os.path.join(
    DATASET_DIR,
    "nuimg-frame_features" if not OOD else "coco-frame_features",
    NUIM_DATASET_VERSION if not OOD else COCO_DATASET_VERSION,
)

# numerical consts
# 0.75 bcs we want really good boxes for features and not mostly good features with 0.5
IOU_THRESH = 0.75
