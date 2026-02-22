import random
import numpy as np
import torch

# config
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 40
SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = True
PERSISTENT_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """
    Each library has its own seed setting function. This function removes the need to hardcode the seeds individually.
    "parts" that need their seeds to be set: random, numpy, torch (and torch.cuda if using GPU)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)