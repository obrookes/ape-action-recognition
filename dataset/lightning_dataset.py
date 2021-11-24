from __future__ import print_function, division
import os
import torch
import random
import json
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage import io, transform
from statistics import mode

from utils.data import *
import pytorch_lightning as pl

"""
Great Ape dataset consisting of frames extracted from jungle trap footage.
Extracts all samples found in raw dataset using provided annotations.
Pre-processes and returns sequences of RGB frames and optical flow images for
the spatial and temporal streams respectively.
"""

class LightningGreatApeDataset(torch.utils.data.Dataset):

    '''
    Args
    data: path to frames
    annotations: path to annotations
    '''

    def __init__(
        self, data, annotations, sample_interval, sequence_length, activity_duration_threshold, jitter, flip, rotation, probability, mode, video_names, classes,
    ):
        super(LightningGreatApeDataset, self).__init__()
        
        # Specifies what split of data this will hold (e.g. train, validation, test)
        self.mode = mode

        # Data sampling parameters
        self.sample_interval = sample_interval
        self.sequence_length = sequence_length
        self.activity_duration_threshold = activity_duration_threshold

        # Obtain from txt file with the names of videos to sample from.
        self.video_names = open(video_names).read().strip().split()

        # Path initialisation
        self.frame_dir = data
        self.annotations_dir = annotations

        self.classes = classes

        # Normalisation and data augmentation transforms
        self.spatial_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ]
        )
        self.temporal_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        (
            self.spatial_augmentation_transform,
            self.temporal_augmentation_transform,
        ) = self.initialise_augmentations(jitter, flip, rotation)
        self.augmentation_probability = probability

        self.labels = 0
        self.samples = {}
        self.samples_by_class = {}

        if self.mode == "train":
            self.initialise_dataset()
            
            self.initialise_labels()
            self.initialise_samples_by_class()
            
            self.labels = torch.from_numpy(self.labels)
        
        elif self.mode == 'triplet':
            self.initialise_dataset()
            
            self.initialise_labels()
            self.initialise_samples_by_class()
            
            self.labels = torch.from_numpy(self.labels)
                    
        elif self.mode == "test" or self.mode == "validation":
            self.initialise_test_dataset()

            self.initialise_labels()
            self.initialise_samples_by_class()
            
            self.labels = torch.from_numpy(self.labels)
        
        elif(self.mode == "inference"):
            self.initialise_inference_dataset()
            
    # Return length of dataset (number of samples)
    def __len__(self):
        # If the length of the dataset has not yet been calculated, then do so and store it
        if not hasattr(self, "size"):
            self.size = 0
            for key in self.samples.keys():
                self.size += len(self.samples[key])

        return self.size

    # Get the ith (index) sample from the dataset
    def __getitem__(self, index):

        # Get details of required sample
        if self.mode == "inference":
            video, ape_id, start_frame = find_sample(self.samples, self.mode, index)
        else:
            video, ape_id, start_frame, activity = find_sample(self.samples, self.mode, index)
        
        # Anchor img
        spatial_sample = self.get_spatial_sample(video, ape_id, start_frame)       
        temporal_sample = self.get_temporal_sample(video, ape_id, start_frame)
        anchor_label = self.classes.index(activity)
        metadata = {"ape_id": ape_id, "start_frame": start_frame, "video": video, "activity": activity} 
        anchor = (spatial_sample, temporal_sample, anchor_label, metadata)

        # Get details of positive sample
        pos_sample = self.get_positive(activity)
        pos = self.get_sample(sample=pos_sample, type='positive')
        
        # Should have some check labels are the same...
        
        neg_sample = self.get_negative(activity)
        neg = self.get_sample(sample=neg_sample, type='negative')
         
        """
        Logic for inference dataset
        """
        
        if(self.mode == "inference"):
            return spatial_sample, temporal_sample, metadata
        elif(self.mode =="triplet"):
            return anchor, pos, neg
        else:
            return spatial_sample, temporal_sample, anchor_label, metadata 


    def get_sample(self, sample, type):

        if(type=='positive'):
        
            pos_spatial = self.get_spatial_sample(sample['video'], sample['ape_id'], sample['start_frame'])       
            pos_temporal = self.get_temporal_sample(sample['video'], sample['ape_id'], sample['start_frame'])
            pos_label = self.classes.index(sample['activity'])
            return (pos_spatial, pos_temporal, pos_label, sample)
        
        elif(type=='negative'):

            neg_spatial = self.get_spatial_sample(sample['video'], sample['ape_id'], sample['start_frame'])       
            neg_temporal = self.get_temporal_sample(sample['video'], sample['ape_id'], sample['start_frame'])
            neg_label = self.classes.index(sample['activity'])
            return (neg_spatial, neg_temporal, neg_label, sample)
        

    def get_positive(self, activity):
        anchor_class = self.samples_by_class[activity]
        pos = random.choice(anchor_class)
        return pos

    
    # Todo: test and finish this...

    def get_negative(self, activity):
        activities = list(self.samples_by_class.keys())
        while(True):
            chosen_activity = random.choice(activities)
            if(chosen_activity != activity):
                if(self.samples_by_class[chosen_activity] != []):
                    break
        neg_class = self.samples_by_class[chosen_activity]
        neg = random.choice(neg_class)
        return neg


    def get_spatial_sample(self, video, ape_id, start_frame):
        """
        Spatial Data
        """
        spatial_sample = []

        # Assign probabilities to activated augmentations
        transform_probabilities = []
        for i in range(0, len(self.spatial_augmentation_transform)):
            transform_probabilities.append(random.random())

        # Build sequence of RGB still frames of size sequence_length
        for i in range(0, self.sequence_length):
            path = f"{self.frame_dir}/rgb/{video}/{video}_frame_{start_frame + i}.jpg"
            
            spatial_image = Image.open(path)

            # Get ape and its coordinates
            ape = get_ape_by_id(self.annotations_dir, video, start_frame + i, ape_id)
            coordinates = get_ape_coordinates(ape)
            
            # Crop around ape
            spatial_image = spatial_image.crop(
                (coordinates[0], coordinates[1], coordinates[2], coordinates[3])
            )

            # Apply augmentation and pre-processing transforms if training
            if self.mode == "train":
                spatial_data = self.apply_augmentation_transforms(
                    spatial_image, self.spatial_augmentation_transform, transform_probabilities
                )
                spatial_data = self.spatial_transform(spatial_data)
            else:
                spatial_data = self.spatial_transform(spatial_image)

            # Add image to sequence
            spatial_sample.append(spatial_data.squeeze_(0))
            spatial_image.close()
                
        # end
        spatial_sample = torch.stack(spatial_sample, dim=0)
        return spatial_sample


    def get_temporal_sample(self, video, ape_id, start_frame):
        """
        Temporal Data
        """
        temporal_sample = []

        # Assign probabilities to activated augmentations
        transform_probabilities = []
        for i in range(0, len(self.temporal_augmentation_transform)):
            transform_probabilities.append(random.random())

        # Build sequence of temporal images of size sequence_length
        for i in range(0, self.sequence_length):
            this_data = []

            # Get optical flow images for both directions
            x_path = f"{self.frame_dir}/horizontal_flow/{video}/{video}_frame_{start_frame + i}.jpg"
            y_path = f"{self.frame_dir}/vertical_flow/{video}/{video}_frame_{start_frame + i}.jpg"


            image_x = Image.open(x_path)
            image_y = Image.open(y_path)

            # Get ape and its coordinates
            ape = get_ape_by_id(self.annotations_dir, video, start_frame + i, ape_id)
            coordinates = get_ape_coordinates(ape)


            # Crop around ape and apply transforms
            image_x = image_x.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))
            image_y = image_y.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))

            # Apply augmentation and pre-processing transforms if training
            if self.mode == "train":
                final_image_x = self.apply_augmentation_transforms(
                    image_x, self.temporal_augmentation_transform, transform_probabilities
                )
                final_image_y = self.apply_augmentation_transforms(
                    image_y, self.temporal_augmentation_transform, transform_probabilities
                )
                final_image_x = self.temporal_transform(final_image_x)
                final_image_y = self.temporal_transform(final_image_y)
            else:
                final_image_x = self.temporal_transform(image_x)
                final_image_y = self.temporal_transform(image_y)

            # Stack both optical flow images in third dimension
            this_data.append(final_image_x.squeeze_(0))
            this_data.append(final_image_y.squeeze_(0))
            stacked_data = torch.stack(this_data, dim=0)

            # Add stacked optical flow image to sequence
            temporal_sample.append(stacked_data.squeeze(0))

            image_x.close()
            image_y.close()
                
        temporal_sample = torch.stack(temporal_sample, dim=0)
        return temporal_sample


    """
    Creates a dictionary which includes all valid spatial + temporal samples from the dataset,
    constrained by parameters provided in config.
    """
    def initialise_dataset(self):

        # Go through every video in dataset
        for video in tqdm(self.video_names, desc=f"Initialising {self.mode} dataset", leave=False):
            # Count how many apes are present in the video
            no_of_apes = get_no_of_apes(self.annotations_dir, video)

            # Go through each ape by id for possible samples
            for current_ape_id in range(0, no_of_apes + 1):

                no_of_frames = len(glob.glob(f"{self.annotations_dir}/{video}/*.xml"))
                frame_no = 1

                # Traverse through every frame to find valid samples
                while frame_no <= no_of_frames:
                    if (no_of_frames - frame_no) < self.activity_duration_threshold:
                        break

                    # Find first instance of ape by id
                    ape = get_ape_by_id(self.annotations_dir, video, frame_no, current_ape_id)

                    if not ape:
                        frame_no += 1
                        continue
                    else:
                        # Check if this ape has the same activity for atleast the next activity_duration_threshold frames
                        current_activity = ape.find("activity").text
                        valid_frames = 1

                        for look_ahead_frame_no in range(frame_no + 1, no_of_frames + 1):
                            ape = get_ape_by_id(
                                self.annotations_dir, video, look_ahead_frame_no, current_ape_id
                            )

                            if (ape) and (ape.find("activity").text == current_activity):
                                valid_frames += 1
                            else:
                                break

                        # If activity spanned over less frames than activity duration threshold, carry on with search
                        if valid_frames < self.activity_duration_threshold:
                            frame_no += valid_frames
                            continue

                        # If this sample meets the required number of frames, break it down into smaller samples with the given interval
                        last_valid_frame = frame_no + valid_frames
                        for valid_frame_no in range(
                            frame_no, last_valid_frame, self.sample_interval
                        ):

                            # For the last valid sample, ensure that there are enough temporal frames with the ape following it
                            if (valid_frame_no + self.sample_interval) >= last_valid_frame:
                                correct_activity = False
                                for temporal_frame in range(valid_frame_no, self.sequence_length):
                                    ape = get_ape_by_id(
                                        self.annotations_dir, video, temporal_frame, current_ape_id
                                    )
                                    ape_activity = get_activity(ape)
                                    if (
                                        (not ape)
                                        or (ape_activity != current_activity)
                                        or (temporal_frame > no_of_frames)
                                    ):
                                        correct_activity = False
                                        break
                                if correct_activity == False:
                                    break

                            # Check if there are enough frames left
                            if (no_of_frames - valid_frame_no) >= self.sequence_length:

                                # Insert sample
                                if video not in self.samples.keys():
                                    self.samples[video] = []

                                # Keep count of number of labels
                                self.labels += 1

                                self.samples[video].append(
                                    {
                                        "ape_id": current_ape_id,
                                        "activity": current_activity,
                                        "start_frame": valid_frame_no,
                                    }
                                )

                        frame_no = last_valid_frame

    
    """
    Creates a dictionary which includes all spatial + temporal samples from the dataset.
    These samples are used for evaluation of data, where no samples are omitted from search.
    """
    def initialise_test_dataset(self):

        # Go through every video in dataset
        for video in tqdm(self.video_names, desc=f"Initialising {self.mode} dataset", leave=False):

            # Count how many apes are present in the video
            no_of_apes = get_no_of_apes(self.annotations_dir, video)

            # Go through each ape by id for possible samples
            for current_ape_id in range(0, no_of_apes + 1):
                no_of_frames = len(glob.glob(f"{self.annotations_dir}/{video}/*.xml"))
                frame_no = 1

                # Traverse through every frame to get samples
                while frame_no <= no_of_frames:
                    if (no_of_frames - frame_no) < (self.sequence_length - 1):
                        break

                    # Find first instance of ape by id
                    ape = get_ape_by_id(self.annotations_dir, video, frame_no, current_ape_id)

                    if not ape:
                        frame_no += 1
                        continue
                    else:
                        activities = []
                        insufficient_apes = False

                        # Check that this ape exists for the next n frames
                        for look_ahead_frame_no in range(frame_no, frame_no + self.sequence_length):
                            ape = get_ape_by_id(
                                self.annotations_dir, video, look_ahead_frame_no, current_ape_id
                            )

                            if ape:
                                activities.append(ape.find("activity").text)
                            else:
                                insufficient_apes = True
                                break

                        # If the ape is not present for enough consecutive frames, then move on
                        if insufficient_apes:
                            # frame_no = look_ahead_frame_no
                            frame_no += self.sequence_length
                            continue

                        # Get majority activity across frames
                        try:
                            activity = mode(activities)
                        except:
                            activity = activities[0]

                        # Check if there are enough frames left
                        if (no_of_frames - frame_no) >= self.sequence_length:

                            # Insert sample
                            if video not in self.samples.keys():
                                self.samples[video] = []

                            # Keep count of number of labels
                            self.labels += 1

                            self.samples[video].append(
                                {
                                    "ape_id": current_ape_id,
                                    "activity": activity,
                                    "start_frame": frame_no,
                                }
                            )

                        frame_no += self.sequence_length

        return
    
    """
    Creates a dictionary which includes all spatial + temporal samples from the dataset.
    These samples are used for evaluation of data, where no samples are omitted from search.
    """
    def initialise_inference_dataset(self):

        # Go through every video in dataset
        for video in tqdm(self.video_names, desc=f"Initialising {self.mode} dataset", leave=False):
            
            # Count how many apes are present in the video
            no_of_apes = get_no_of_apes(self.annotations_dir, video)
            
            
            # Go through each ape by id for possible samples
            for current_ape_id in range(0, no_of_apes + 1):
                no_of_frames = len(glob.glob(f"{self.annotations_dir}/{video}/*.xml"))
                frame_no = 1

                # Traverse through every frame to get samples
                while frame_no <= no_of_frames:
                    if (no_of_frames - frame_no) < (self.sequence_length - 1):
                        break

                    # Find first instance of ape by id
                    ape = get_ape_by_id(self.annotations_dir, video, frame_no, current_ape_id)

                    if not ape:
                        frame_no += 1
                        continue
                    else:
                        activities = []
                        insufficient_apes = False

                        # Check that this ape exists for the next n frames
                        for look_ahead_frame_no in range(frame_no, frame_no + self.sequence_length):
                            ape = get_ape_by_id(
                                self.annotations_dir, video, look_ahead_frame_no, current_ape_id
                            )

                            if ape:
                                continue
                            else:
                                insufficient_apes = True
                                break

                        # If the ape is not present for enough consecutive frames, then move on
                        if insufficient_apes:
                            # frame_no = look_ahead_frame_no
                            frame_no += self.sequence_length
                            continue

                        # Check if there are enough frames left
                        if (no_of_frames - frame_no) >= self.sequence_length:

                            # Insert sample
                            if video not in self.samples.keys():
                                self.samples[video] = []

                            # Keep count of number of labels
                            self.labels += 1

                            self.samples[video].append(
                                {
                                    "ape_id": current_ape_id,
                                    "start_frame": frame_no,
                                }
                            )

                        frame_no += self.sequence_length

        return

    # Creates a dictionary which includes all spatial + temporal samples from the dataset
    # with the classes as keys.
    def initialise_samples_by_class(self):
        for class_name in self.classes:
            self.samples_by_class[class_name] = []

        for video in self.samples.keys():
            for annotation in self.samples[video]:
                new_annotation = annotation
                new_annotation["video"] = video
                self.samples_by_class[annotation["activity"]].append(new_annotation)

    # Returns a list of number of samples for each class
    def get_no_of_samples_by_class(self):
        samples_by_class = np.unique(self.labels.cpu().numpy(), return_counts=True)[1]
        return samples_by_class

    # Initialise class label for each sample
    def initialise_labels(self):
        labels = np.zeros(self.labels, dtype=int)
        i = 0

        for video in self.samples:
            for annotation in self.samples[video]:
                label = self.classes.index(annotation["activity"])
                labels[i] = label
                i += 1

        self.labels = labels

    # Return label by index
    def _get_label(self, index):
        return self.labels[index]

    # Add augmentations to be considered for training according to config
    def initialise_augmentations(self, jitter, flip, rotation):

        spatial_aug = []
        temporal_aug = []

        if jitter:
            spatial_aug.append(
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            )
        if flip:
            spatial_aug.append(transforms.RandomHorizontalFlip(p=1))
        if rotation:
            spatial_aug.append(transforms.RandomRotation(p=1, degrees=10))

        
        # if augmentation_cfg['temporal']['horizontal_flip']:
            # temporal_aug.append(transforms.RandomHorizontalFlip(p=1))

        return spatial_aug, temporal_aug

    # Apply augmentations to image depending on probability
    def apply_augmentation_transforms(self, image, augmentations, probabilities):
        for i, aug in enumerate(augmentations):
            if probabilities[i] < self.augmentation_probability:
                image = aug(image)

        return image
