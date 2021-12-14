import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
from dataset.dataset import LightningGreatApeDataset

class PanAfDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, 
            batch_size, 
            num_workers,
            sample_interval,
            seq_length,
            behaviour_threshold,
            compute
            ):
        
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_interval = sample_interval
        self.seq_length = seq_length
        self.behaviour_threshold = behaviour_threshold
        self.compute = compute

        if(self.compute=='local'):
            self._FRAMES = '/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/data/frames'
            self._ANNOTATIONS = '/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/data/annotations'
            self._TRAIN_VIDEOS = '/home/dl18206/Desktop/phd/code/personal/pan-africa-annotation/action-recognition/splits/traindata.txt'
            self._VAL_VIDEOS = '/home/dl18206/Desktop/phd/code/personal/pan-africa-annotation/action-recognition/splits/valdata.txt'
            self._TEST_VIDEOS = '/home/dl18206/Desktop/phd/code/personal/pan-africa-annotation/action-recognition/splits/testdata.txt'
            self._CLASSES = open('classes.txt').read().strip().split()
        
        elif(self.compute=='hpc'):
            self._FRAMES = '/mnt/storage/scratch/dl18206/frames'
            self._ANNOTATIONS = '/mnt/storage/scratch/dl18206/annotations'
            self._TRAIN_VIDEOS = '/mnt/storage/scratch/dl18206/splits/train.txt'
            self._VAL_VIDEOS = '/mnt/storage/scratch/dl18206/splits/val.txt'
            self._TEST_VIDEOS = '/mnt/storage/scratch/dl18206/splits/test.txt'
            self._CLASSES = open('/mnt/storage/scratch/dl18206/classes.txt').read().strip().split()


    def train_dataloader(self):

        train_dataset = LightningGreatApeDataset(
            data=self._FRAMES,
            annotations=self._ANNOTATIONS,
            sample_interval=self.sample_interval,
            sequence_length=self.seq_length,
            activity_duration_threshold=self.behaviour_threshold,
            jitter=False,
            flip=False,
            rotation=False,
            probability=0,
            mode='train',
            video_names=self._TRAIN_VIDEOS,
            classes=self._CLASSES
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        
       val_dataset = LightningGreatApeDataset(
            data=self._FRAMES,
            annotations=self._ANNOTATIONS,
            sample_interval=self.sample_interval,
            sequence_length=self.seq_length,
            activity_duration_threshold=self.behaviour_threshold,
            jitter=False,
            flip=False,
            rotation=False,
            probability=0,
            mode='validation',
            video_names=self._VAL_VIDEOS,
            classes=self._CLASSES
        )

       return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
            )
