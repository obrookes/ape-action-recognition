import random
import torch
import torchmetrics
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import pytorchvideo.models.resnet
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.transforms import MixUp, AugMix, CutMix
from dataset.datamodule import PanAfDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.plugins import DDPPlugin
from kornia.losses import FocalLoss

class VideoClassificationLightningModule(pl.LightningModule):
  
    def __init__(self, model_name, loss, alpha, gamma, optimiser, freeze_backbone, learning_rate, momentum, weight_decay, augmentation, augmentation_probability):
      super().__init__()

      
      self.model_name = model_name

      if(loss == "focal"):
              self.loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
      if(loss == "cross_entropy"):
              self.loss = nn.CrossEntropyLoss()

      self.optimiser = optimiser
      self.freeze_backbone = freeze_backbone
    
      # Load pretrained model
      pretrained_model = torch.hub.load("facebookresearch/pytorchvideo:main", model=self.model_name, pretrained=True)
      
      # Strip the head from backbone  
      self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

      # Attach a new head with specified class number (hard coded for now...)
      self.res_head = create_res_basic_head(
              in_features=2048, out_features=500
      )

      self.fc = nn.Linear(in_features=500, out_features=9)
      
      # Dropout hardcoded 0 for now
      self.dropout = nn.Dropout(p=0)

      if self.freeze_backbone:
          for param in self.backbone.parameters():
              param.requires_grad = False

      self.aug_method = augmentation

      self.learning_rate = learning_rate
      self.momentum = momentum
      self.weight_decay = weight_decay
      
      if(self.aug_method=='mixup'):
          self.augmentation = MixUp(num_classes=9)
      elif(self.aug_method=='augmix'):
          self.augmentation = AugMix()
      elif(self.aug_method=='cutmix'):
          self.augmentation = CutMix(num_classes=9)
      else:
          self.augmentation = None

      self.augmentation_probability = augmentation_probability
      
      # Metric initialisation
      self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
      self.top3_train_accuracy = torchmetrics.Accuracy(top_k=3)
      self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)  
      self.top3_val_accuracy = torchmetrics.Accuracy(top_k=3)

    def forward(self, x):
        output = self.dropout(self.res_head(self.backbone(x)))
        return self.fc(output)

    def training_step(self, batch, batch_idx):
      
      # The model expects a video tensor of shape (B, C, T, H, W)
      data, label, meta = batch

      if(random.random() <= self.augmentation_probability):
          if(self.aug_method == 'mixup' or self.aug_method == 'cutmix'):
              data, label = self.augmentation.forward(data, label)
              label = label.max(dim=1).indices
          elif(self.aug_method == 'augmix'):
              data_T = data.transpose(dim0=1, dim1=2)
              data = torch.stack([self.augmentation(v) for v in data_T], dim=0).transpose(dim0=1, dim1=2)
          else:
              raise ValueError(f'Trying to apply non-existant augmentation')          
      
      pred = self(data)
      loss = self.loss(pred, label)
      
      top1_train_acc = self.top1_train_accuracy(pred, label)
      top3_train_acc = self.top3_train_accuracy(pred, label)
      
      self.log('top1_train_acc', top1_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False)
      self.log('top3_train_acc', top3_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False) 
      self.log('train_loss', loss.item(), logger=True, on_epoch=True, on_step=True)

      return {"loss": loss, "logs": {"train_loss": loss.detach(), "top1_train_acc": top1_train_acc, "top3_train_acc": top3_train_acc}}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_train_accuracy.compute()
        top3_acc = self.top3_train_accuracy.compute()
        self.log('train_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)   

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
 

    def validation_step(self, batch, batch_idx):
  
      data, label, meta = batch
      pred = self(data)
      loss = F.cross_entropy(pred, label)
      
      top1_val_acc = self.top1_val_accuracy(pred, label)
      top3_val_acc = self.top3_val_accuracy(pred, label) 

      self.log('top1_val_acc', top1_val_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False)
      self.log('top3_val_acc', top3_val_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False)
      self.log('val_loss', loss, logger=True, on_epoch=True, on_step=False)
      
      return {"loss": loss, "logs": {"val_loss": loss.detach(), "top1_val_acc": top1_val_acc, "top3_val_acc": top3_val_acc}}

    def validation_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_val_accuracy.compute()
        top3_acc = self.top3_val_accuracy.compute()
        self.log('val_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)  

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
 
    def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      if(self.optimiser=='sgd'):
          optimiser=torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
      elif(self.optimiser=='adam'):
          optimiser=torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
      else:
          raise ValueError('Unknown argument passed to --optimiser')
      
      return {
              "optimizer": optimiser,
              "lr_scheduler": {
                  "scheduler": ReduceLROnPlateau(optimizer=optimiser, mode="max", patience=5, verbose=True),
                  "monitor": "val_top1_acc_epoch",
                  "frequency": 1
            },
        }
    
    def get_lr(self):
        return self.learning_rate

def main(args):
    
    # Input all needs to come for argparse eventually...
    classification_module = VideoClassificationLightningModule(model_name='slow_r50',
            loss=args.loss,
            alpha=args.alpha,
            gamma=args.gamma,
            optimiser=args.optimiser,
            freeze_backbone=args.freeze_backbone,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            augmentation=args.augmentation,
            augmentation_probability=args.augmentation_prob
            )
    
    data_module = PanAfDataModule(batch_size=args.batch_size,
            num_workers = args.num_workers,
            sample_interval = args.sample_interval,
            seq_length = args.seq_length,
            behaviour_threshold = args.behaviour_threshold,
            balanced_sampling=args.balanced_sampling,
            compute = args.compute
            )
    
    # Checkpoint callbacks    
    val_acc_checkpoint = ModelCheckpoint(
        monitor="val_top1_acc_epoch",
        dirpath=args.save_ckpt,
        filename="top1_acc_{epoch}",
        mode="max"
    )
    
    tb_logger = loggers.TensorBoardLogger('log', name='behaviour_recognition')

    if(args.gpus > 0):
        trainer = pl.Trainer(callbacks=[val_acc_checkpoint],
                        replace_sampler_ddp=False,
                        gpus=args.gpus, 
                        num_nodes=args.nodes,
                        strategy=DDPPlugin(find_unused_parameters=True),
                        precision=16,
                        accumulate_grad_batches=args.acc_batches,
                        stochastic_weight_avg=args.swa, 
                        max_epochs=args.epochs) 
    else:    
        trainer = pl.Trainer(auto_lr_find=True) 

    # Tune for optimum lr
    # trainer.tune(classification_module, data_module)

    # Train
    trainer.fit(classification_module, data_module)

if __name__== "__main__":
   
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Trainer args - specify nodes and GPUs
    parser.add_argument('--compute', type=str, required=True,
            help='Specify either "local" or "hpc"')
    parser.add_argument('--gpus', type=int, default=0, required=False,
            help='Specify the number of GPUs per node for training. Default is 0 (i.e. train on CPU)')
    parser.add_argument('--nodes', type=int, default=1, required=False,
            help='Specify the number of nodes used in training. Default is 0')
    
    # Training config - sampling
    parser.add_argument('--batch_size', type=int, required=True, 
            help='Specify the batch size per iteration of training')
    parser.add_argument('--acc_batches', type=int, required=False, default=1,
            help='Specify number of batches to accumulate')
    parser.add_argument('--balanced_sampling', type=str, default=None,
            help='Specify "balanced" or "dynamic". The default is None.')
    parser.add_argument('--num_workers', type=int, required=True,
            help='Specify the number of workers')

    # Training config - loss
    parser.add_argument('--loss', type=str, default='cross_entropy', required=False,
            help='Specify loss function i.e. "focal" or "cross_entropy". Default is "cross_entropy"')    
    parser.add_argument('--alpha', type=float, default=1, required=False)
    parser.add_argument('--gamma', type=float, default=2, required=False)

    # Training config - optimiser
    parser.add_argument('--optimiser', type=str, default='sgd', required=False,
            help='Specify optimiser i.e. "sgd" or "adam". Default is "sgd"')

    # Training config - fine-tuning
    parser.add_argument('--freeze_backbone', type=int, required=True,
            help='Specify whether to freeze layers EXCEPT the final layer for fine-tuning')
    
    # Training config = other hparams
    parser.add_argument('--learning_rate', type=float, default=0.0001, required=False)
    parser.add_argument('--momentum', type=float, default=0, required=False)
    parser.add_argument('--weight_decay', type=float, default=0, required=False)

    parser.add_argument('--swa', type=int, required=False, default=1, 
            help='Enable stochastic weight averaging (swa). Default is 1 (True)')

    parser.add_argument('--augmentation', type=str, default=None, 
            help='Specify type of augmentation i.e. MixUp or AugMix. Default is None')
    parser.add_argument('--augmentation_prob', type=float, default=0.5, 
            help='Specify the probability at which augmention is applied')

    # Training config - epochs
    parser.add_argument('--epochs', type=int, default=10, required=False,
            help='Specify the total number of training epochs')
    
    # Dataset configuration
    parser.add_argument('--sample_interval', type=int, default=20, 
            help='The interval between consecutive frames to sample. Default is 20')
    parser.add_argument('--seq_length', type=int, default=5,
            help='The length of the sequence to sample. Default is 5')
    parser.add_argument('--behaviour_threshold', type=int, default=72,
            help='The length of time (in frames) a behaviour must be exhibited to be a valid sample at training time. Default is 72')

    # Path where ckpt file is saved
    parser.add_argument('--save_ckpt', type=str, required=True,
            help='Specify path where model checkpoint should be saved')

    args = parser.parse_args()

    main(args)
