import os
import torch
from ignite.engine import Engine,Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import CosineAnnealingScheduler,create_lr_scheduler_with_warmup,ProgressBar
from pytorch_pretrained_bert import cached_path
from torch.distributed.fsdp import FullyShardedDataParalle as FSDP
import sys,tqdm
from model import StableCAT
from transformers import GPT2Tokenizer,BertTokenizer,AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    VOCAB_SIZE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEADS,
    NUM_LAYER,
    grad_norm_clip,
)

#encode text sequence using any encoder of choice
# Any BPETokenizer is supported 
# tokenizer = Tokenizer.from_file('/peregrine_tokenizer.json')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if tokenizer.pad_token is None:
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#load dataset and tokenize with the tokenizer

dataset_cache = './Dataset/dataset'
if os.path.isfile(dataset_cache):
  dataset=torch.load(dataset_cache)
else:
  dataset_file = cached_path('https://s3.amazonaws.com/datasets.huggingface.co/wikitext-v2/wiki.train.tokens')
  with open(dataset_file,'r',encoding='utf-8') as f:
    dataset = f.readlines()
  dataset = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
    line.strip(' ').replace('\n','[SEP]').replace('<unk>','[UNK]')
  )) for line in tqdm.tqdm(dataset))
  dataset = torch.tensor([index for line in dataset for index in line],dtype=torch.long)
  torch.save(dataset,dataset_cache)
num_sequences = (dataset.size(0) // 128) * 128
dataset = dataset.narrow(0,0,num_sequences).view(-1,128)
batch_size = 32
data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

#instantiate the model
model = StableCAT(
    vocab_size=VOCAB_SIZE,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,)

m = model.to(DEVICE)
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)
print('model with {:.2f}M parameters'.format(sum(p.numel() for p in m.parameters()) / 1e6))


#add FSDP for parallel training
#ensure device is set to cuda before enabling FSDP
def enable_fsdp(model):
    shared_module = FSDP(model)
    return shared_module

''' to add FSDP: m = enable_fsdp(model), then add model to device . Always remember to call enable_fsdp before initializing the optimizer'''

def train(engine,batch) -> None:
  m.train()
  batch = batch.transpose(0,1).contiguous()
  input_ids = batch
  input_ids = input_ids.to(DEVICE)
  logits ,loss= m(input_ids,input_ids)
  loss = loss / 4
  loss.backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(),grad_norm_clip)
  if engine.state.iteration % 4 == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
  return loss.item()

#Training is done using an awesome library called pytorch igniter
trainer = Engine(train)
RunningAverage(output_transform=lambda x: x).attach(trainer,'loss')
ProgressBar(persist=True).attach(trainer,metric_names=['loss'])
cos_scheduler = CosineAnnealingScheduler(optimizer,'lr',LEARNING_RATE,0.0,len(data_loader) * N_EPOCHS)
scheduler = create_lr_scheduler_with_warmup(cos_scheduler,0.0,3,N_WARMUP)
trainer.add_event_handler(Events.ITERATION_STARTED,scheduler)
checkpoint_handler = ModelCheckpoint('PATH TO CKPT HANDLER','checkpoint',save_interval=1,n_saved=5)
trainer.add_event_handler(Events.EPOCH_COMPLETED,checkpoint_handler,{'StableCAT':m})
torch.save(m.state_dict(),os.path.join('PATH TO SAVE','stable.bin'))
trainer.run(data_loader,max_epochs=N_EPOCHS)

#Run inference with saved checkpoints
checkpoints = "PATH TO SAVED CHECKPOINT"
m.load_state_dict(torch.load(checkpoints))
m.eval()
inputs = tokenizer('SOME INPUTS')

load = torch.tensor(inputs.input_ids,dtype=torch.long)
# this is done to reshape the input size for cropping 
load = load.unsqueeze(0)
print(load)
ans = m.generate(load,100BLOCK_SIZE)[0]
ans = ans.tolist()

print(tokenizer.decode(ans))
