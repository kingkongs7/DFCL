'''
# author: Yulin Yao
# email: yuliny@bupt.edu.cn
# date: 2025-05-18
'''

import sys
sys.path.append('.')

import os
import cv2
import lmdb
import torch
import random
import math

import numpy as np
import albumentations as A
import torchvision.transforms as T

from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial.distance import pdist
from scipy.fft import fft2, fftshift
from torch.utils.data import Sampler
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset

def process_image(image_path, lmdb_path, block_size=16):
    env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        # transfer the path format from rgb-path to lmdb-key
        if image_path[0]=='.':
            image_path=image_path.replace('./datasets\\','')

        image_bin = txn.get(image_path.encode())
        image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    feats = []
    for i in range(64, h-64, block_size):
        for j in range(64, w-64, block_size):
            blk = gray[i:i+block_size, j:j+block_size]
            if blk.shape == (block_size, block_size):
                # b = (blk.astype(np.float32) - blk.mean()) / (blk.std() + 1e-6)
                F = fftshift(fft2(blk))
                feats.append(np.log1p(np.abs(F)).flatten())

    if len(feats) < 2:
        return None

    dists = pdist(feats, metric='euclidean')
    return dists.var()

class SlideDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        '''
        self.image_list = []
        self.label_list = []
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        '''
        print('calculate var...')
        self.stats_list = []
        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
        tasks = [
            (img_path, lmdb_path, 16) for img_path in self.image_list
        ]
        with Pool(processes=20) as pool:
            for result in tqdm(pool.starmap(process_image, tasks), total=len(self.image_list)):
                if result is not None:
                    self.stats_list.append(result)
        print('calculate var done...')

        # all_indices = np.arange(len(self.label_list))
        # real_idx = all_indices[np.array(self.label_list) == 0]
        # fake_idx = all_indices[np.array(self.label_list) == 1]
        
        # fake_stats = np.array(self.stats_list)[fake_idx]
        # median_val = np.median(fake_stats)
        # # median_val = np.percentile(fake_stats, 30)
        # keep_fake_idx = fake_idx[fake_stats <= median_val]
        
        # keep_idx = np.concatenate([real_idx, keep_fake_idx])
        # keep_idx.sort()  
        
        # self.stats_list  = [self.stats_list[i]  for i in keep_idx]
        # self.image_list  = [self.image_list[i]  for i in keep_idx]
        # self.label_list  = [self.label_list[i]  for i in keep_idx]

        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation

        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask
    

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                # image = self.detect_and_crop_face(image, mode='test')
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors

class SlideSampler(Sampler):
    def __init__(self, label_list, stats_list, total_epochs=10, T0=1, T1=5, initial_ratio=0.4, final_ratio=1.0):
        super().__init__(None)
        self.labels = np.array(label_list)
        self.stats  = np.array(stats_list, dtype=float)
        self.total_epochs = total_epochs
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.T0 = T0
        self.T1 = T1

        # init index
        self.real_indices = np.where(self.labels == 0)[0]
        self.fake_indices = np.where(self.labels == 1)[0]
        self.current_epoch = 0
        self.losses = None

    def _compute_gamma(self):
        beta = 0.8
        if self.current_epoch <= self.T0:
            return beta
        elif self.current_epoch <= self.T1:
            return beta - 0.5*(self.current_epoch - self.T0)/(self.T1 - self.T0)
        else:
            return beta - 0.5
        # return 1.0

    def update_state(self, epoch, losses):
        self.current_epoch = epoch
        self.losses = np.array(losses)

    def __iter__(self):
        if self.losses  is None:
            raise RuntimeError("Losses not updated! Call update_state() first.")
        
        normalized_stats = (self.stats-np.min(self.stats)) / (np.max(self.stats) - np.min(self.stats) + 1e-6)
        normalized_dynamic = self.losses / (np.max(self.losses) + 1e-6)

        _gamma = self._compute_gamma()
        selection_scores = (1 - normalized_stats) * _gamma + (1 - _gamma) * (1 - normalized_dynamic)
        
        if self.current_epoch <= self.T0:
            fake_scores = selection_scores[self.fake_indices]
            n_select = int(len(self.fake_indices) * self.initial_ratio)
            selected_fake = self.fake_indices[np.argsort(-fake_scores)[:n_select]] 
            
            indices = np.concatenate([self.real_indices,  selected_fake])
            
        elif self.current_epoch  <= self.T1:
            current_ratio = self.initial_ratio + (1-self.initial_ratio) * (self.current_epoch-self.T0) / (self.T1-self.T0)
            selected_count = int(len(self.labels)*current_ratio)
            indices = np.argsort(-selection_scores)[:selected_count] 
            
        else:
            weights = np.exp(selection_scores) 
            weights /= weights.sum() 
            indices = np.random.choice(len(self.labels),  len(self.labels),  p=weights, replace=False)
        
        np.random.shuffle(indices) 
        return iter(indices.tolist()) 

    def __len__(self):
        return len(list(self.__iter__()))