import os
import cv2
import json
from scipy.stats.stats import mode
import torch
import csv
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torchaudio
import torchaudio.transforms as audio_T
import pdb
import time
from PIL import Image, ImageFilter
import glob
import sys
import scipy.io.wavfile as wav
from scipy import signal
import random
import pickle
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

torchaudio.set_audio_backend("sox_io")
sys.path.append('./datasets/')


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
 
        data = []
        self.args = args

        if args.dataset_mode == 'VGGSound':
            args.trainset_path = '/GPFS/public/VGGSound/'
        elif args.dataset_mode == 'Flickr':
            args.trainset_path = '/GPFS/public/Flickr-SoundNet/'

        # Debug with a small dataset
        if args.debug:
            
            if mode=='train':
                with open('metadata/debug_data/train_vggss_debug_100.txt','r') as f:
                    txt_reader = f.readlines()
                    for item in txt_reader:
                        data.append(item.rstrip('\n'))
                    self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                    self.video_path = args.trainset_path + '/total_video_frames/'
            
            elif mode=='test':
                with open('metadata/debug_data/test_vggss_debug_50.txt','r') as f:
                    txt_reader = f.readlines()
                    for item in txt_reader:
                        data.append(item.split('.')[0])
                    self.audio_path = args.vggss_test_path + '/audio/'
                    self.video_path = args.vggss_test_path + '/frame/'

            elif mode=='val':
                with open('metadata/test_flick.csv') as f:
                    # if arg.test == 'test.csv':
                    csv_reader = csv.reader(f)
                    for item in csv_reader:
                        data.append(item[0])
                    
                    self.audio_path = args.soundnet_test_path + '/audio/'
                    self.video_path = args.soundnet_test_path + '/frames/'

        else:
            if args.dataset_mode == 'VGGSound':
                if mode=='train':
                    if self.args.training_set_scale == 'subset_144k':
                        train_list_file = 'train_vggss_train_144k.txt'
                    elif self.args.training_set_scale == 'subset_10k':
                        train_list_file = 'train_vggss_train_10k.txt'
                    else:
                        train_list_file = 'train_vggss_190228.txt' 

                    with open('metadata/' + train_list_file,'r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.rstrip('\n'))
                        self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                        self.video_path = args.trainset_path + '/total_video_frames/'

                elif mode=='test':
                    with open('metadata/test_vggss_4911.txt','r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.split('.')[0])
                        self.audio_path = args.vggss_test_path + '/audio/'
                        self.video_path = args.vggss_test_path + '/frame/'
                
                elif mode=='val':
                    with open('metadata/test_flick.csv') as f:
                        # if arg.test == 'test.csv':
                        csv_reader = csv.reader(f)
                        for item in csv_reader:
                            data.append(item[0])
                        
                        self.audio_path = args.soundnet_test_path + '/audio/'
                        self.video_path = args.soundnet_test_path + '/frame/'
                    
           
            elif args.dataset_mode == 'Flickr':
                if mode == 'train':
                    if self.args.training_set_scale == 'subset_10k':
                        train_list_file = 'train_flickr_10k.txt'
                    else:
                        train_list_file = 'train_flickr_144k.txt'
                    with open('metadata/' + train_list_file,'r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.rstrip('\n'))
                        self.audio_path = args.trainset_path + '/mp3/'
                        self.video_path = args.trainset_path + '/frames/'
                
                elif mode == 'test':
                    with open('metadata/test_flickr_250.txt','r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.split('.')[0])
                        self.audio_path = args.soundnet_test_path + '/mp3/'
                        self.video_path = args.soundnet_test_path + '/frame/'
                

        self.imgSize = args.image_size 
        self.AmplitudeToDB = audio_T.AmplitudeToDB()
        self.mode = mode
        self.transforms = transforms
        self._init_atransform()
        self._init_transform()
        #Load the similar audio pairs dict
        self._load_audio_pairs()

        #  Retrieve list of audio and video files
        self.video_files = []
   
        for item in data[:]:
            self.video_files.append(item )

        print("{0} dataset size: {1}".format(self.mode.upper() , len(self.video_files)))
        
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        if self.mode == 'train':

            if self.args.img_aug == 'moco_v1':
                augmentation = [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]

                self.img_transform = transforms.Compose(augmentation)

            elif self.args.img_aug == 'moco_v2':
                augmentation = [
                    transforms.RandomResizedCrop(224, scale=(0.3, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
                
                self.img_transform = transforms.Compose(augmentation)
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])            

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])
    
    def _load_audio_pairs(self):
        pairs_dict_path = os.path.join(self.args.retri_save_dir, 'pairs_'+str(self.args.epoch)+'.pkl' )
        if os.path.isfile(pairs_dict_path):
            with open(pairs_dict_path, 'rb') as f:
                pairs_dict = pickle.load(f)
            self.simi_pairs_dict = pairs_dict
        else:
            self.simi_pairs_dict = None


    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]
        if self.args.dataset_mode == 'VGGSound':
            if self.mode == 'train':
                frame = self.img_transform(self._load_frame(os.path.join( self.video_path, file , '125.jpg' ) ))
                frame_ori = np.array(self._load_frame(os.path.join( self.video_path, file , '125.jpg' ) ))
                if (self.simi_pairs_dict is not None)  and (random.random() < self.args.audio_mix_prob):
                    file_1 = self.simi_pairs_dict[file]
                    samples1, samplerate1 = torchaudio.load(os.path.join(self.audio_path, file_1 + '.wav'))
                    samples0, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))
                    mixed_len = min(samples0.size(-1), samples1.size(-1))
                    samples = samples0[...,:mixed_len] * (1-self.args.audio_mix_alpha) + \
                                samples1[...,:mixed_len] * self.args.audio_mix_alpha 


                else:
                    samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))

            elif self.mode in ['test', 'val'] :
                frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '.jpg')  ))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))
        
        elif self.args.dataset_mode == 'Flickr':
            if self.mode == 'train':
                frame = self.img_transform(self._load_frame(os.path.join(self.video_path, file, '00000003.jpg')))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file, '00000003.jpg') ))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.mp3'))
                # Only the first four seconds of the audio is used, when training
                samples = samples[...,:samplerate * 4]

            elif self.mode in ['test', 'val'] :
                frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '.jpg')  ))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.mp3'))
  

        if samples.shape[1] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[1]) + 1
            samples = samples.repeat(1, n)
        # import ipdb; ipdb.set_trace()
        samples = samples[...,:samplerate*10]

        spectrogram  =  audio_T.MelSpectrogram(
                sample_rate=samplerate,
                n_fft=512,
                hop_length=239,  # 512-273
                n_mels=257,
                normalized=True

            )(samples)
        
        if (self.args.aud_aug=='SpecAug') and (self.mode=='train') and (random.random() < 0.8):
            maskings = nn.Sequential(
                audio_T.TimeMasking(time_mask_param=180),
                audio_T.FrequencyMasking(freq_mask_param=35)
                )
            spectrogram = maskings(spectrogram)

        spectrogram = self.AmplitudeToDB(spectrogram)


        return frame, spectrogram, 'samples', file, torch.tensor(frame_ori)



