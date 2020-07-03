import json
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os

import torch.nn as nn
from torchvision import transforms 
from processData import Vocabulary
from encoder import Encoder
from decoder import Decoder
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer, BertModel
from data_loader import get_loader
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

from config import Config

from caption import caption_image_beam_search




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Config()
criterion = nn.CrossEntropyLoss().to(device)

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = True)

def load_image(image_path, transform=None):
    try:
        image = Image.open(image_path)
        image = image.resize([224, 224], Image.LANCZOS)
        
        if transform is not None:
            image = transform(image).unsqueeze(0)
        
        return image
    except Exception as e:
         print(f"Error at image {image_path}; {e}")
         return image

def get_val_images(val_path):
    f = open(config.validation_path, 'r+', encoding='utf8')
    data = json.load(f)
    return data['images']

def main():
    print("Initializing...")
    config = Config()
     # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("Vocabulary loaded")

    #Build models
    encoder = Encoder().eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Decoder(vocab_size=len(vocab),use_glove=False, use_bert=False, vocab=vocab, device=device, BertTokenizer=tokenizer, BertModel=BertModel)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print("Model built")
    encoder_path = args.encoder_path if args.encoder_path else config.encoder_path
    decoder_path = args.decoder_path if args.decoder_path else config.decoder_path
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    print("Model loaded")
    images = get_val_images(config.validation_path)

    print("Validation file loaded")
    results_data = []
    curr_id = 0

    for index, image_data in enumerate(images):
        try:
            image_path = config.val_img_path + "/" + image_data['file_name']
            image = load_image(image_path, transform= transform)
            image_tensor = image.to(device)

            caption_idx, _ = caption_image_beam_search(encoder = encoder, decoder = decoder, word_map = vocab, image = image)
            print(f"Caption index: {caption_idx}")
        except Exception as e:
            print(e)
            pass

main()