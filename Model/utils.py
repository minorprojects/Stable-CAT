import os
import torch
from datetime import datetime
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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


#encode text sequence using peregrine_tokenizer
# Any BPETokenizer is supported sunce the peregrine tokenizer is trained using BPE(Byte Pair encoding)
def tokenize_dataset_func(examples, tokenizer: any,data_sub):
    return tokenizer.encode(examples[data_sub])

def tokenizer_decode(enc_sec: torch.Tensor,tokenizer:any):
    text = tokenizer.decode(enc_sec)
    return text

#you can use the pytorcch dataLoader class , which is what i also used during some part of the training
def get_batch(data: list[str],block_size: int,batch_size: int):
    ix = torch.randint(len(data) - block_size,(batch_size))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x,y = x.to(DEVICE), y.to(DEVICE)
    return x,y
    
#add FSDP for parallel training
#ensure device is set to cuda before enabling FSDP
def enable_fsdp(model):
    shared_module = FSDP(model)
    return shared_module

@torch.no_grad()
def estimate_loss(
    data: list[str],
    model: torch.nn.Module,
    block_size: int,
    batch_size: int,
    eval_iters: int = 10,
    ):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X,Y = get_batch(data=data,block_size=block_size,batch_size=batch_size)
        logits,loss = model.forward(X,Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

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
    model:torch,nn.Module,path_to_checkpoint:str = "checkpoints", epoch:int=0):
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
   
