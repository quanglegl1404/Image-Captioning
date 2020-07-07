'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script loads the COCO dataset in batches to be used for training/testing
''' 

import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from config import Config

config = Config()

class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        #print(f"Image id: {img_id}")
        path = coco.loadImgs(img_id)[0]['file_name']
        fullPath = os.path.join(self.root, path);
        #print(f"Full path: {fullPath}")
        try:
            ##todo: pass
            # if os.path.exists(fullPath):
            #     print(f"{fullPath} does exist")
            image = Image.open(fullPath).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            #print("Done tokenizing")
            caption = []
            caption.append(vocab('<start>'))
            #print("Start extending caption")
            for token in tokens:
<<<<<<< HEAD
=======
                #print(f"token: {token}")
>>>>>>> a7461e6cedbc041fe00145c20a5960eda91a727c
                if token == '<unk>':
                    caption.extend([vocab['[UNK]']])
                else:
                    try:
                        caption.extend([vocab(token)])
                    except:
                        print(f"Error extending token {token}")
                        pass
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)

            return image, target, img_id

        except Exception as e:
            print(f"Exception: {e}")
            print(f"Image path: {fullPath}")
            return None, None, None

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    
    if(data == None):
        print("Data is empty")
        return

    data.sort(key=lambda  x: len(x[1]), reverse=True)
    #print(data)
    images, captions, img_ids = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, img_ids

def get_loader(method, vocab, batch_size):

    # train/validation paths
    if method == 'train':
        root = config.train_img_path
        json = config.caption_path
    elif method =='val':
        root = '../../Images/resized2017'
        json = config.validation_path

    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader