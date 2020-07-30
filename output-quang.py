import json
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from itertools import groupby 

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

criterion = nn.CrossEntropyLoss().to(device)

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = True)

class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    f = open(val_path, 'r+', encoding='utf8')
    data = json.load(f)
    return data['images']

def sample(i, imgs, caps, caplens, decoder, encoder):

    imgs_jpg = imgs.numpy() 
    imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)

    losses = loss_obj()
    
    # Forward prop.
    imgs = encoder(imgs.to(device))
    caps = caps.to(device)

    seq, _ = decoder.sample(imgs)
    return seq

    # scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
    # targets = caps_sorted[:, 1:]

    # # Remove timesteps that we didn't decode at, or are pads
    # scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
    # targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

    # # Calculate loss
    # loss = criterion(scores_packed, targets_packed)
    # loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
    # losses.update(loss.item(), sum(decode_lengths))

    # # Hypotheses
    # _, preds = torch.max(scores, dim=2)
    # preds = preds.tolist()
    # temp_preds = list()
    # for j, p in enumerate(preds):
    #     pred = p[:decode_lengths[j]]
    #     pred = [w for w in pred if w not in [config.PAD, config.START, config.END]]
    #     temp_preds.append(pred)  # remove pads, start, and end
    # preds = temp_preds
    
    # return preds

def main():
    print("Initializing...")

     # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    print("Load vocab")
    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print("Get val loader")
    val_loader = get_loader('val', vocab, config.batch_size)

    print("Vocabulary loaded")
    #Build models
    encoder = Encoder().eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Decoder(vocab_size=len(vocab),use_glove=False, use_bert=config.bert_model, vocab=vocab, device=device, BertTokenizer=tokenizer, BertModel=BertModel)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print("Model built")
    encoder_path = config.encoder_path
    decoder_path = config.decoder_path
    encoder.load_state_dict(torch.load(encoder_path), strict=False)
    decoder.load_state_dict(torch.load(decoder_path), strict=False)

    images = get_val_images(config.validation_path)

    print("Valdation file loaded")

    results_data = []
    results_img = set()
    curr_id = 0
    for i, (imgs, caps, caplens, ids) in enumerate(tqdm(val_loader)):
        try:
            #image_path = config.val_img_path + imgs['file_name']
            #image = load_image(image_path, transform)
            #image_tensor = image.to(device)

            # Generate an caption from the image
            sampled_ids = sample(i, imgs, caps, caplens, decoder, encoder)
            print(sampled_ids)
            #sampled_ids = sampled_ids[]        # (1, max_seq_length) -> (max_seq_length)

            # for j, word_array in enumerate(sampled_ids):
            #     img_id = ids[j]
            #     if img_id not in results_img:
            #         results_img.add(img_id)
            #         sampled_caption = []
            #         token_list = []
            #         word_array = [i[0] for i in groupby(word_array)]
            #         for word_id in word_array:
            #             token = vocab.idx2word[word_id]
            #             token_list.append(token)
            #             # sampled_caption.append(word)
            #         sentence = ' '.join(token_list)

            #         print(f"{sentence}")
            #         record = {
            #             'image_id': img_id,
            #             'caption': sentence,
            #             'id': curr_id
            #         }
            #         curr_id+=1

            #         results_data.append(record)

        except Exception as e:
            print(e)
            pass

    # with open(config.machine_output_path, 'w+') as f_results:
    #     f_results.write(json.dumps(results_data, ensure_ascii=False))
    
    print("Finished")
    #         sampled_caption = []

    #         for word_id in sampled_ids:
    #             word = vocab.idx2word[word_id]
    #             if not word in ['<start>', '<end>']:
    #                 sampled_caption.append(word)
    #             if word == '<end>':
    #                 break
    #         sentence = ' '.join(sampled_caption)
    #         sentence = sentence.replace('<start> ','')
    #         sentence = sentence.replace(' <end>', '')
    #         record = {
    #             'image_id': int(imgs['id']),
    #             'caption': sentence,
    #             'id': curr_id
    #         }
    #         curr_id+=1
    #         results_data.append(record)
    #         if index%10 == 0:
    #             print(f"Done image {index}/{len(images)}")
    #     except Exception as e:
    #         print(e)
    #         pass
    # with open(config.machine_output_path, 'w+') as f_results:
    #     f_results.write(json.dumps(results_data, ensure_ascii=False))
    # print("Finished")
    # Print out the image and the generated caption
    # print (sentence)
    # image = Image.open(path)
    # plt.imshow(np.asarray(image))

if __name__ == "__main__":
    main()