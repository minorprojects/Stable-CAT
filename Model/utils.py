import os
import torch
from datetime import datetime
#some hyperparameters, feel free to modify these to scale the model parameters 
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
grad_norm_clip = 1.0
BLOCK_SIZE =128
VOCAB_SIZE=50304,
NUM_EMBED=512,
NUM_HEADS=8,
NUM_LAYER=12,
DROPOUT=0.1,
N_EPOCHS = 500
N_WARMUP = 1000
#I assume you know what this does, it's just selecting device(either cuda if available or cpu) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#some handy functions to perfrom minor operations, although this might not be so useful
def tokenize_dataset_func(examples):
    return tokenizer(examples['text'],padding='max_length',truncation=True)

def tokenizer_decode(enc_sec: torch.Tensor):
    text = tokenizer.decode(enc_sec)
    return text


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
   
