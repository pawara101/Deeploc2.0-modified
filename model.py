import math
import pytorch_lightning as pl
from esm import Alphabet, FastaBatchedDataset, pretrained
from transformers import T5EncoderModel, T5Tokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attr_prior import *
from pathlib import Path
import pkg_resources

class AttentionHead(nn.Module):
      def __init__(self, hidden_dim, n_heads):
          super(AttentionHead, self).__init__()
          self.n_heads = n_heads
          self.hidden_dim = hidden_dim
          self.preattn_ln = nn.LayerNorm(hidden_dim//n_heads)
          self.Q = nn.Linear(hidden_dim//n_heads, n_heads, bias=False)
          torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1/(hidden_dim//n_heads))

      def forward(self, x, np_mask, lengths):
          # input (batch, seq_len, embed)
          n_heads = self.n_heads
          hidden_dim = self.hidden_dim
          x = x.view(x.size(0), x.size(1), n_heads, hidden_dim//n_heads)
          x = self.preattn_ln(x)
          mul = (x * \
                self.Q.weight.view(1, 1, n_heads, hidden_dim//n_heads)).sum(-1) \
                #* np.sqrt(5)
                #/ np.sqrt(hidden_dim//n_heads)
          mul_score_list = []
          for i in range(mul.size(0)):
              # (1, L) -> (1, 1, L) -> (1, L) -> (1, L, 1)
              mul_score_list.append(F.pad(smooth_tensor_1d(mul[i, :lengths[i], 0].unsqueeze(0), 2).unsqueeze(0),(0, mul.size(1)-lengths[i]),"constant").squeeze(0))
          
          mul = torch.cat(mul_score_list, dim=0).unsqueeze(-1)
          mul = mul.masked_fill(~np_mask.unsqueeze(-1), float("-inf"))
          
          attns = F.softmax(mul, dim=1) # (b, l, nh)
          x = (x * attns.unsqueeze(-1)).sum(1)
          x = x.view(x.size(0), -1)
          return x, attns.squeeze(2)

class ProtT5Frozen(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.initial_ln = nn.LayerNorm(1024)
        self.lin = nn.Linear(1024, 256)
        self.attn_head = AttentionHead(256, 1)
        self.clf_head = nn.Linear(256, 11)

    def forward(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_attns

    def predict(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_pool, x_attns

class ESM1bFrozen(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.initial_ln = nn.LayerNorm(1280)
        self.lin = nn.Linear(1280, 256)
        self.attn_head = AttentionHead(256, 1)
        self.clf_head = nn.Linear(256, 11)

    def forward(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_attns

    def predict(self, embedding, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.initial_ln(embedding)
        x = self.lin(x)
        x_pool, x_attns = self.attn_head(x, non_mask, lens)
        x_pred = self.clf_head(x_pool)
        #print(x_pred, x_attns)
        return x_pred, x_pool, x_attns

class SignalTypeMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(267, 32)
        self.ln2 = nn.Linear(32, 9)
        self.lr = 1e-3

    def forward(self, x):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = nn.Tanh()(self.ln1(x))
        x = self.ln2(x)
        return x

class ProtT5E2E(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_func = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval()
        self.subcel_clfs = nn.ModuleList([ProtT5Frozen.load_from_checkpoint(pkg_resources.resource_filename("DeepLoc2",f"models/models_prott5/{i}_1Layer.ckpt"), map_location="cpu").eval() for i in range(5)])
        self.signaltype_clfs = nn.ModuleList([SignalTypeMLP.load_from_checkpoint(pkg_resources.resource_filename("DeepLoc2",f"models/models_prott5/signaltype/{i}.ckpt"), map_location="cpu").eval() for i in range(5)])

    def forward(self, toks, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        x = self.embedding_func(input_ids=torch.tensor(toks['input_ids'], device=self.device),
            attention_mask=torch.tensor(toks['attention_mask'], 
                device=self.device)).last_hidden_state[:, :-1].float()
        x_loc_preds, x_signal_preds, x_attnss = [], [], []
        for i in range(5):
          x_pred, x_pool, x_attns = self.subcel_clfs[i].predict(x, lens.to(self.device), non_mask[:, :-1].to(self.device))
          x_loc_preds.append(torch.sigmoid(x_pred))
          x_attnss.append(x_attns)
          x_signal_pred = torch.sigmoid(self.signaltype_clfs[i](torch.cat((x_pool, torch.sigmoid(x_pred)), dim=1)))
          x_signal_preds.append(x_signal_pred)

        return torch.stack(x_loc_preds).mean(0).cpu().numpy(), torch.stack(x_attnss).mean(0).cpu().numpy(), torch.stack(x_signal_preds).mean(0).cpu().numpy()

class ESM1bE2E(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model, alphabet = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
        self.embedding_func = model.eval()
        self.subcel_clfs = nn.ModuleList([ESM1bFrozen.load_from_checkpoint(pkg_resources.resource_filename(__name__,f"models/models_esm1b/{i}_1Layer.ckpt"), map_location="cpu").eval() for i in range(5)])
        self.signaltype_clfs = nn.ModuleList([SignalTypeMLP.load_from_checkpoint(pkg_resources.resource_filename(__name__,f"models/models_esm1b/signaltype/{i}.ckpt"), map_location="cpu").eval() for i in range(5)])

    def forward(self, toks, lens, non_mask):#, dct_mat, idct_mat):
        # in lightning, forward defines the prediction/inference actions
        device = self.device
        x = self.embedding_func(toks.to(self.device), repr_layers=[33])["representations"][33][:, 1:-1].float()
        x_loc_preds, x_signal_preds, x_attnss = [], [], []
        for i in range(5):
          x_pred, x_pool, x_attns = self.subcel_clfs[i].predict(x, lens.to(self.device), non_mask[:, 1:-1].to(self.device))
          x_loc_preds.append(torch.sigmoid(x_pred))
          x_attnss.append(x_attns)
          x_signal_pred = torch.sigmoid(self.signaltype_clfs[i](torch.cat((x_pool, torch.sigmoid(x_pred)), dim=1)))
          x_signal_preds.append(x_signal_pred)

        return torch.stack(x_loc_preds).mean(0).cpu().numpy(), torch.stack(x_attnss).mean(0).cpu().numpy(), torch.stack(x_signal_preds).mean(0).cpu().numpy()
