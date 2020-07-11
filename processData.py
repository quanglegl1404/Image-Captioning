'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script processes the COCO dataset
'''  

import os
import pickle
from collections import Counter
import nltk
from PIL import Image
from pycocotools.coco import COCO
from config import Config
#from vncorenlp import VnCoreNLP

#rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['[unk]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold, tokenizer):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = tokenize(caption.lower(),tokenizer)
        for j, item in enumerate(tokens):
            #print(item)
            if("_" in item):
                #print(True)
                temp = tokenize(item.replace("_", " "),tokenizer)
                counter.update(temp)
            else:
                counter.update(tokens)
        # if("_" in enumerate(tokens))
        #     tokens.replace("_", " ")

        #counter.update(tokens)

        if (i+1) % 100000 == 0:
            print(f"[{i+1}/{len(ids)}] Tokenized the captions.")
    # ommit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('[pad]') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('[unk]') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def tokenize(caption, tokenizer):
    # if tokenizer == 'rdr':
    #     return rdrsegmenter.tokenize(caption)[0]
    
    return nltk.tokenize.word_tokenize(caption)

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main():
    config = Config()
    caption_path = config.caption_path
    threshold = config.threshold
    vocab_path = config.vocab_path

    vocab = build_vocab(json=caption_path,threshold=threshold, tokenizer=config.tokenizer)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    #print(vocab.__dict__)
    print(f"Vocabulary size: {len(vocab)}")

    # print("resizing images...")
    # splits = ['val','train']

    # for split in splits:
    #     folder = './data/%s2017' %split
    #     resized_folder = './data/%s2017_resized/' %split
    #     if not os.path.exists(resized_folder):
    #         os.makedirs(resized_folder)
    #     image_files = os.listdir(folder)
    #     num_images = len(image_files)
    #     for i, image_file in enumerate(image_files):
    #         with open(os.path.join(folder, image_file), 'r+b') as f:
    #             with Image.open(f) as image:
    #                 image = resize_image(image)
    #                 image.save(os.path.join(resized_folder, image_file), image.format)

    # print("done resizing images...")

#caption_path = './data/annotations/uitviic_captions_val2017.json'
vocab_path = './data/vocab.pkl'
threshold = 1

main()