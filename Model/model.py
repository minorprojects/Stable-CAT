import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
#attention head for peregrine

class AttentionHead(nn.Module):
    def __init__(self,num_embed,head_size,block_size,dropout):
        super(AttentionHead,self).__init__()
        self.q = nn.Linear(num_embed,head_size,bias=False)
        self.k = nn.Linear(num_embed,head_size,bias=False)
        self.v = nn.Linear(num_embed,head_size,bias=False)
        
        #create lower triangular matrix for masking
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        B, T , C = x.shape
      #rotary embedding from th roformer paper
        rotary_emb = RotaryEmbedding(dim=96)
        key = rotary_emb.rotate_queries_or_keys(self.k(x))
        query = rotary_emb.rotate_queries_or_keys(self.q(x))
        value = self.v(x)
        outs = None 

        #add flash attention for awesomeness
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(
                    query,key,value,
                    attn_mask = mask,
                    dropout_p = flash_attn_dropout,
                    is_casual=casual,
                    scale=scale)
                print('using flash attention')
                outs = out
        if torch.backends.cuda.flash_sdp_enabled():
            return outs
        else:
            attn = query @ key.transpose(-2,-1) * C** -0.5
            #c = dim(k)
            attn = attn.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
            attn = F.softmax(attn,dim=-1)
            print('flash attention disabled') 
            outs = attn @ value
            return outs
#multi head attention class
class MHA(nn.Module):
    def __init__(self,num_heads,head_size,num_embed,block_size,dropout):
        super(MHA,self).__init__()
        self.heads = nn.ModuleList(
        [
            AttentionHead(head_size=head_size,
                         num_embed=num_embed,
                         block_size=block_size,
                         dropout=dropout
                         )
            for i in range(num_heads)
        ])
        self.proj = nn.Linear(num_embed,num_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class MLP(nn.Module):
    def __init__(self,num_embed,dropout):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(num_embed,4 * num_embed),
        nn.GELU(),
        nn.Linear(4 * num_embed,num_embed),
        nn.Dropout(dropout)
        )
        
    def forward(self,x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,num_heads,block_size,num_embed,dropout):
        super().__init__()
        head_size = num_embed // num_heads
        self.sa = MHA(
            num_heads=num_heads,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffn = MLP(num_embed=num_embed,dropout=dropout)
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class StableCAT(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.vocab_size = kwargs.get('vocab_size',100)
        self.num_embed = kwargs.get('num_embed',32)
        self.block_size = kwargs.get('block_size',8)
        self.num_heads = kwargs.get('num_heads',4)
        self.num_layers = kwargs.get('num_layers',4)
        self.dropout = kwargs.get('dropout', 0.2)
        self.token_embedding = nn.Embedding(self.vocab_size,self.num_embed)
        self.position_embedding = nn.Embedding(self.block_size,self.num_embed)
        self.blocks = nn.Sequential(
        
            *[
                TransformerBlock(
                num_heads=self.num_heads,
                block_size=self.block_size,
                num_embed=self.num_embed,
                dropout=self.dropout,
                )
                for _ in range(self.num_layers)
                
            ]
        )
        self.ln_f = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)
        
    def forward(self,idx,targets=None):
        B, T = idx.shape
        token_embed = self.token_embedding(idx)
        posit_embed = self.position_embedding(torch.arange(T,device=DEVICE))
        x = token_embed + posit_embed
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets != None:
            B,T,C = logits.shape
            logits = torch.reshape(logits,(B * T,C))
            targets = torch.reshape(targets,(B * T))
            loss = F.cross_entropy(logits,targets)
        else:
            loss = None
        return logits, loss
    #courtsey to Andrej Karpathyh nanoGPT for the generate function, i think openAi did something similar
    def generate(self,idx: torch.Tensor,max_new_tokens: int,block_size: int):
        for _ in range(max_new_tokens):
            #crop the context to the last block_size tokens
            idx_crop = idx[:, -block_size:]
            logits ,loss = self.forward(idx_crop)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
            
        return idx
 
