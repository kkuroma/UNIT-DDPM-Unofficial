import torch

#image
IMG_SIZE = 64
ANIME_PATH = "/anime/"
HUMAN_PATH = "/human/"

#diffusion
TIMESTEPS = 1000
RELEASE_TIME = 100

#hyperparams
DIM = 32
LR = 1e-5
BATCH_SIZE = 32

#training strategy
EPOCHS = 100
LOAD_FROM_CHECKPOINT = False
CHECKPOINT_DIR = "/"

#other
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")