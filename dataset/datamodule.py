import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
from dataset.dataset import LightningGreatApeDataset

class PanAfDataModule(pytorch_lightning.LightningDataModule):

    # Dataset configuration
    
    _FRAMES = '/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/data/frames'
    _ANNOTATIONS = '/home/dl18206/Desktop/phd/code/personal/ape-behaviour-triplet-network/data/annotations'
    _TRAIN_VIDEOS = '/home/dl18206/Desktop/phd/code/personal/pan-africa-annotation/action-recognition/splits/traindata.txt'
    _VAL_VIDEOS = '/home/dl18206/Desktop/phd/code/personal/pan-africa-annotation/action-recognition/splits/valdata.txt'
    _TEST_VIDEOS = '/home/dl18206/Desktop/phd/code/personal/pan-africa-annotation/action-recognition/splits/testdata.txt'
    _CLASSES = open('classes.txt').read().strip().split()

    # Dataloader configuration
    _BATCH_SIZE = 1
    _NUM_WORKERS = 6  # Number of parallel processes fetching data

    def train_dataloader(self):

        train_dataset = LightningGreatApeDataset(
            data=self._FRAMES,
            annotations=self._ANNOTATIONS,
            sample_interval=5,
            sequence_length=5,
            activity_duration_threshold=10,
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
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        
       val_dataset = LightningGreatApeDataset(
            data=self._FRAMES,
            annotations=self._ANNOTATIONS,
            sample_interval=5,
            sequence_length=5,
            activity_duration_threshold=10,
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
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            )
