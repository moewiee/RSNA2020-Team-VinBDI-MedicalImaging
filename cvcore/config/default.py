from yacs.config import CfgNode as CN


# Create root config node
_C = CN()
# Config name
_C.NAME = ""
# Config version to manage version of configuration names and default
_C.VERSION = "0.1"


# ----------------------------------------
# System config
# ----------------------------------------
_C.SYSTEM = CN()

# Number of workers for dataloader
_C.SYSTEM.NUM_WORKERS = 8
# Use GPU for training and inference. Default is True
_C.SYSTEM.CUDA = True
# Random seed for seeding everything (NumPy, Torch,...)
_C.SYSTEM.SEED = 0
# Use half floating point precision
_C.SYSTEM.FP16 = True
# FP16 Optimization level. See more at: https://nvidia.github.io/apex/amp.html#opt-levels
_C.SYSTEM.OPT_L = "O2"


# ----------------------------------------
# Directory name config
# ----------------------------------------
_C.DIRS = CN()

# Train, Validation and Testing image folders
_C.DIRS.TRAIN_IMAGES = ""
_C.DIRS.VALIDATION_IMAGES = ""
_C.DIRS.TEST_IMAGES = ""
# Trained weights folder
_C.DIRS.WEIGHTS = "./weights/"
# Inference output folder
_C.DIRS.OUTPUTS = "./outputs/"
# Training log folder
_C.DIRS.LOGS = "./logs/"
_C.DIRS.EMBEDDINGS = "./embeddings/"


# ----------------------------------------
# Datasets config
# ----------------------------------------
_C.DATA = CN()

# Create small subset to debug
_C.DATA.DEBUG = False
# Datasets problem (multiclass / multilabel)
_C.DATA.TYPE = ""
# Image size for training
_C.DATA.IMG_SIZE = (224, 224)
# Image input channel for training
_C.DATA.INP_CHANNEL = 3
# For CSV loading dataset style
# If dataset is contructed as folders with one class for each folder, see ImageFolder dataset style
# Train, Validation and Test CSV files
_C.DATA.CSV = ""
_C.DATA.TEST_CSV = ""
# For ImageFolder dataset style
# TODO #
# Dataset augmentations style (albumentations / randaug / augmix)
_C.DATA.AUGMENT = ""
# For randaug augmentation. For augmix or albumentations augmentation, refer to those other section
_C.DATA.RANDAUG = CN()
# Number of augmentations picked for each iterations. Default is 2
_C.DATA.RANDAUG.N = 2
# Amptitude of augmentation transform (0 < M < 30). Default is 27
_C.DATA.RANDAUG.M = 27
# Use ranged amptitude for augmentations transforms. Default is False.
_C.DATA.RANDAUG.RANDOM_MAGNITUDE = False
# For augmix augmentaion
_C.DATA.AUGMIX = CN()
_C.DATA.AUGMIX.ALPHA = 1.
_C.DATA.AUGMIX.BETA = 1.
# For albumentations augmentation
# TODO #
# Cutmix data transformation for training
_C.DATA.CUTMIX = CN({"ENABLED": False})
_C.DATA.CUTMIX.ALPHA = 1.0
# Mixup data transformation for training
_C.DATA.MIXUP = CN({"ENABLED": False})
_C.DATA.MIXUP.ALPHA = 1.0
# Gridmask data transformation for training
_C.DATA.GRIDMASK = CN({"ENABLED": False})


# ----------------------------------------
# Training config
# ----------------------------------------
_C.TRAIN = CN()

# Number of training cycles
_C.TRAIN.NUM_CYCLES = 1
# Number of epoches for each cycle. Length of epoches list must equals number of cycle
_C.TRAIN.EPOCHES = [50]
# Training batchsize
_C.TRAIN.BATCH_SIZE = 32


# ----------------------------------------
# Inference config
# ----------------------------------------
_C.INFER = CN()
# Save prediction
_C.INFER.SAVE_NAME = ""


# ----------------------------------------
# Solver config
# ----------------------------------------
_C.SOLVER = CN()

# Solver algorithm
_C.SOLVER.OPTIMIZER = "adamw"
# Solver scheduler (constant / step / cyclical)
_C.SOLVER.SCHEDULER = "cyclical"
# Warmup length. Set 0 if do not want to use
_C.SOLVER.WARMUP_LENGTH = 0
# Use gradient accumulation. If not used, step equals 1
_C.SOLVER.GD_STEPS = 1
# Starting learning rate (after warmup, if used)
_C.SOLVER.BASE_LR = 1e-3
# Weight decay coeffs
_C.SOLVER.WEIGHT_DECAY = 1e-2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
# Stochastic weights averaging
_C.SOLVER.SWA = CN({"ENABLED": False})
# SWA starting epoch
_C.SOLVER.SWA.START_EPOCH = 10
# SWA update frequency (iterations)
_C.SOLVER.SWA.FREQ = 5
# SWA decay coeff for moving average
_C.SOLVER.SWA.DECAY = 0.999


# ----------------------------------------
# Loss function config
# ----------------------------------------
_C.LOSS = CN()

# Loss function (ce / focal / dice)
_C.LOSS.NAME = "ce"


# ----------------------------------------
# Model config
# ----------------------------------------
_C.MODEL = CN()

# Classification model arch
_C.MODEL.NAME = "resnet50"
# Load ImageNet pretrained weights
_C.MODEL.PRETRAINED = True
# Classification head
_C.MODEL.CLS_HEAD = 'linear'
# Number of classification class
_C.MODEL.NUM_CLASSES = 1
# Pooling method (adaptive pooling or generalized mean pooling)
_C.MODEL.POOL = "adaptive_pooling"
# Dropout factor in training
_C.MODEL.DROPOUT = 0.
# Drop path factor for EfficientNet
_C.MODEL.DROPPATH = 0.
# Use hypercolumns
_C.MODEL.HYPER = False

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`