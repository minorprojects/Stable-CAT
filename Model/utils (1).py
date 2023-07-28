import os
import torch
from datetime import datetime
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer
#some hyperparameters
MAX_ITER = 5000
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
betas = (0.9,0.95)
WEIGHT_DECAY = 0.1
grad_norm_clip = 1.0
BLOCK_SIZE =100
VOCAB_SIZE=12000,
NUM_EMBED=512,
NUM_HEADS=96,
NUM_LAYER=96,
DROPOUT=0.1,

#i assume you know what this does, it's just selecting device(either cuda if available or cpu) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def load_checkpoint(
    model_class: torch.nn.Module,
    path_to_checkpoint: str = '',
    **kwargs: dict,) -> torch.nn.Module:
    try:
        state_dict = torch.load(path_to_checkpoint)
        print('successfully loaded model')
    except Exception as e:
        print(f"Error loading model from checkpoint.{e}")
        
    model = model_class(**kwargs)
    model.load_state_dict(state_dict)
    return model

def save_model_to_checkpoint(
    model:torch.nn.Module,path_to_checkpoint:str = "checkpoints", epoch:int=0):
    if not os.path.exists(path_to_checkpoint):
        os.makedirs(path_to_checkpoint)
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H:%S")
    checkpoint_name = 'checkpoint_epoch-' + str(epoch) + "_" + dt_string + '.pt'
    full_path = os.path.join(path_to_checkpoint, checkpoint_name)
    try:
        torch.save(model.state_dict(),full_path)
        print('successfully saved the model to {}'.format(full_path))
    except Exception as e:
        print(f"error saving the model to checkpoint.{e}")
   
