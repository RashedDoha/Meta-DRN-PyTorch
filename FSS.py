import os
import random
from config import data_config as cfg
from utils import download_file_from_google_drive

import torch
from torch.utils.data import Dataset
import numpy as np

class FSSDataset(Dataset):
    folder = cfg['dataset_dir']
    def __init__(self,root,ways,shots=1,test_shots=1,meta_split='train', transform=None, download=True):
        super(FSSDataset, self).__init__()
        assert meta_split in ['train','val','test'], "meta-split must be either 'train', 'val' or 'test'"
        self.root = root
        self.ways = ways
        self.shots = shots
        self.transform = transform
        self.test_shots = test_shots
        self.meta_split = meta_split

        self.root = os.path.expanduser(os.path.join(self.root,self.folder))
        os.makedirs(self.root, exist_ok=True)
        if download:
            self.download()
        all_classes = os.listdir(self.root)

        if meta_split == 'train':
            self.classes = [all_classes[i] for i in range(cfg['n_train_classes'])]
        elif meta_split == 'val':
            self.classes = [all_classes[i] for i in range(cfg['n_train_classes'],cfg['n_train_classes']+cfg['n_val_classes'])]
        else:
            self.classes =  [all_classes[i] for i in range(cfg['n_train_classes']+cfg['n_val_classes'],cfg['n_classes'])]
        self.num_classes = len(self.classes)

    def make_batch(self,class_idx,split):
        shots = self.shots if split=='train' else self.test_shots
        img_batch = torch.zeros((shots,self.ways,3,cfg['img_size'],cfg['img_size']))
        mask_batch = torch.zeros((shots,self.ways,3,cfg['img_size'],cfg['img_size']))
        if split == 'train':
            indices = [(class_idx+i)%10 for i in range(1,self.ways+1)]
            random.shuffle(indices)
        else:
            indices = [random.choice(range(1,11)) for i in range(self.ways)]
        for j in range(shots):
            images = torch.zeros((self.ways,3,cfg['img_size'],cfg['img_size']))
            masks = torch.zeros((self.ways,3,cfg['img_size'],cfg['img_size']))
            for c,i in zip(range(class_idx,class_idx+self.ways), indices):
                c = c%self.num_classes
                img_path = os.path.join(self.root,self.classes[c],str(i)+'.jpg')
                mask_path = os.path.join(self.root, self.classes[c],str(i)+'.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(img_path).convert('RGB')
                transformed = self.transform(image=np.array(img),mask=np.array(mask))
                print(transformed['mask'].shape)
                images[c-class_idx,:,:,:],masks[c-class_idx,:,:,:] = transformed['image'], transformed['mask']
                
            img_batch[j,:,:,:,:] = images
            mask_batch[j,:,:,:,:] = masks
        return img_batch, mask_batch

    def __getitem__(self,index):
        class_idx = int(index/10)
        train_img_batch = self.make_batch(class_idx,'train')
        test_img_batch = self.make_batch(class_idx,'test')
        return {'train':train_img_batch, 'test':test_img_batch}

    def __len__(self):
        return self.num_classes*10

    def download(self,remove_zip=True):
        filename = os.path.join(self.root, cfg['dataset_dir']+'.zip')
        import zipfile
        import tarfile
        import shutil


        file_id = cfg['gdrive_file_id']
        if os.path.isfile(filename):
            return
        
        download_file_from_google_drive(file_id, filename)

        with zipfile.ZipFile(filename, 'r') as f:
            f.extractall(self.root)
        if remove_zip:
            os.remove(filename)

    

        
        




    



    


