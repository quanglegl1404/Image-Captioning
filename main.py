'''
Parts of this code were incorporated from the following github repositories:
1. parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script has the Encoder and Decoder models and training/validation scripts. 
Edit the parameters sections of this file to specify which models to load/run
''' 

# coding: utf-8

import _pickle as pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from processData import Vocabulary
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import skimage.transform
from PIL import Image
import matplotlib.image as mpimg
from torchtext.vocab import Vectors, GloVe
from scipy import misc
from pytorch_pretrained_bert import BertTokenizer, BertModel
import imageio
from encoder import Encoder
from decoder import Decoder
from config import Config
import os

###################
# START Parameters
###################

config = Config()
# hyperparams
grad_clip = config.grad_clip
num_epochs = config.num_epochs
batch_size = config.batch_size
decoder_lr = config.decoder_lr

# if both are false them model = baseline

glove_model = config.glove_model
bert_model = config.bert_model

from_checkpoint = False
train_model = True
valid_model = False

###################
# END Parameters
###################

# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = True)

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
BertModel.eval()


# vocab indices
PAD = config.PAD
START = config.START
END = config.END
UNK = config.UNK

# Load vocabulary
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# load data
train_loader = get_loader('train', vocab, batch_size)

#############
# Init model
#############

criterion = nn.CrossEntropyLoss().to(device)

else:
    encoder = Encoder().to(device)
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                                 lr=1e-4)
    decoder = Decoder(vocab_size=len(vocab),use_glove=glove_model, use_bert=bert_model, vocab=vocab, device=device, BertTokenizer=tokenizer, BertModel=BertModel).to(device)

    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)

###############
# Train model
###############

def train():
    print("Create directory for checkpoints")
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)


    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        min_losses = 5.0
        try:
            for i, (imgs, caps, caplens, _) in enumerate(tqdm(train_loader)):

                if imgs == None:
                    print('Skip none')
                    continue

                imgs = encoder(imgs.to(device))
                caps = caps.to(device)

                scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

                targets = caps_sorted[:, 1:]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                loss = criterion(scores, targets).to(device)

                loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

                decoder_optimizer.zero_grad()
                if encoder_optimizer is not None:
                    encoder_optimizer.zero_grad()
                loss.backward()

                # grad_clip decoder
                for group in decoder_optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)

                decoder_optimizer.step()
                if encoder_optimizer is not None:
                    encoder_optimizer.step()
                losses.update(loss.item(), sum(decode_lengths))

                # save model each 50 batches
                if i%50==0 and i!=0:
                    print('epoch '+str(epoch+1)+'/60 ,Batch '+str(i)+'/'+str(num_batches)+' loss: '+str(losses.avg))
                        # adjust learning rate (create condition for this)
                    for param_group in decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8

            print(f'Loss: {loss}')
            torch.save(decoder.state_dict(),'./checkpoints/decoder_'+str(epoch+1)+'_july_31_vie_bert.ckpt')
            torch.save(decoder.state_dict(),'./checkpoints/encoder_'+str(epoch+1)+'_july_31_vie_bert.ckpt')

            print('epoch checkpoint saved')
        except Exception as ex:
            print(ex)
            pass

    print("Completed training...")  



######################
# Run training/validation
######################

if train_model:
    train()
