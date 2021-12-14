import torch
import torchmetrics
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorchvideo.models.resnet
from pytorchvideo.models.head import create_res_basic_head
from dataset.datamodule import PanAfDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.plugins import DDPPlugin

class VideoClassificationLightningModule(pl.LightningModule):
  
    def __init__(self, model_name, freeze_backbone, learning_rate):
      super().__init__()


      self.model_name = model_name
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
      
      self.dropout = nn.Dropout(p=0.25)

      if self.freeze_backbone:
          for param in self.backbone.parameters():
              param.requires_grad = False
     
      self.learning_rate = learning_rate

      # Metric initialisation
      self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
      self.top3_train_accuracy = torchmetrics.Accuracy(top_k=3)
      self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)  
      self.top3_val_accuracy = torchmetrics.Accuracy(top_k=3) 

    def forward(self, x):
        output = self.dropout(self.res_head(self.backbone(x)))
        return self.fc(output)

    def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the
      # format provided by the dataset
      data, label, meta = batch
      pred = self(data)

      loss = F.cross_entropy(pred, label)
      
      top1_train_acc = self.top1_train_accuracy(pred, label)
      top3_train_acc = self.top3_train_accuracy(pred, label)
        
      probs = F.softmax(pred, dim=1)
      
      print(f"Probs: {probs}")
      print(f"Label: {label}")
        
      train_mAP = torchmetrics.functional.average_precision(probs, label, num_classes=9, average='macro')
      
      print(f"mAP: {train_mAP}")

      self.log('top1_train_acc', top1_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=True)
      self.log('top3_train_acc', top3_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False)
      self.log('train_mAP', train_mAP, logger=True, on_epoch=False, on_step=True, prog_bar=False)
      self.log('train_loss', loss.item(), logger=True, on_epoch=True, on_step=True)

      return {"loss": loss, "logs": {"train_loss": loss.detach(), "top1_train_acc": top1_train_acc, "top3_train_acc": top3_train_acc, "train_mAP": train_mAP}}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_train_accuracy.compute()
        top3_acc = self.top3_train_accuracy.compute()
        self.log('train_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)
        self.log('train_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)  

        # Log mAP
        train_mAP_epoch = torch.stack([x['logs']['train_mAP'] for x in outputs]).mean()
        self.log('train_mAP_epoch', train_mAP_epoch, logger=True, on_epoch=True, on_step=False, prog_bar=False) 

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
 

    def validation_step(self, batch, batch_idx):
  
      data, label, meta = batch
      pred = self(data)
      loss = F.cross_entropy(pred, label)
      
      top1_val_acc = self.top1_val_accuracy(pred, label)
      top3_val_acc = self.top3_val_accuracy(pred, label)

      probs = F.softmax(pred, dim=1)
      val_mAP = torchmetrics.functional.average_precision(probs, label, num_classes=9, average='macro') 

      self.log('top1_val_acc', top1_val_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False)
      self.log('top3_val_acc', top3_val_acc, logger=False, on_epoch=False, on_step=True, prog_bar=False)
      self.log('val_mAP', val_mAP, logger=True, on_epoch=False, on_step=True, prog_bar=False)
      self.log('val_loss', loss, logger=True, on_epoch=True, on_step=False)
      
      return {"loss": loss, "logs": {"val_loss": loss.detach(), "top1_val_acc": top1_val_acc, "top3_val_acc": top3_val_acc, "val_mAP": val_mAP}}

    def validation_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_val_accuracy.compute()
        top3_acc = self.top3_val_accuracy.compute()
        self.log('val_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)  

        # Log mAP
        val_mAP_epoch = torch.stack([x['logs']['val_mAP'] for x in outputs])
        self.log('val_mAP_epoch', val_mAP_epoch, logger=True, on_epoch=True, on_step=False, prog_bar=False)

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=False)
 
    def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def get_lr(self):
        return self.learning_rate

def main(args):
    
    # Input all needs to come for argparse eventually...
    classification_module = VideoClassificationLightningModule(model_name='slow_r50',
            freeze_backbone=args.freeze_backbone,
            learning_rate=args.learning_rate)
    
    data_module = PanAfDataModule(batch_size=args.batch_size,
            num_workers = args.num_workers,
            sample_interval = args.sample_interval,
            seq_length = args.seq_length,
            behaviour_threshold = args.behaviour_threshold,
            compute = args.compute
            )
    
    # Checkpoint callbacks    
    val_acc_checkpoint = ModelCheckpoint(
        monitor="val_top1_acc_epoch",
        dirpath="/mnt/storage/scratch/dl18206/behaviour_recognition/acc",
        filename="best_validation_acc_epoch={epoch}",
        mode="max"
    )
    
    val_mAP_checkpoint = ModelCheckpoint(
        monitor="val_mAP_epoch",
        dirpath="/mnt/storage/scratch/dl18206/behaviour_recognition/mAP",
        filename="best_validation_mAP_epoch={epoch}",
        mode="max"
    )

    tb_logger = loggers.TensorBoardLogger('log', name='behaviour_recognition')

    if(args.gpus > 0):
        trainer = pl.Trainer(callbacks=[val_acc_checkpoint, val_mAP_checkpoint],
                        gpus=args.gpus, 
                        num_nodes=args.nodes,
                        strategy=DDPPlugin(find_unused_parameters=True),
                        precision=16,
                        min_epochs=args.epochs,
                        auto_lr_find=True) 
    else:    
        trainer = pl.Trainer(auto_lr_find=True) 

    # Tune for optimum lr
    trainer.tune(classification_module, data_module)

    # Train
    trainer.fit(classification_module, data_module)

if __name__== "__main__":
   
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Running locally or in the cloud
    parser.add_argument('--compute', type=str, required=True,
            help='Specify either "local" or "hpc"')

    # Specify nodes and GPUs
    parser.add_argument('--gpus', type=int, default=0, required=False,
            help='Specify the number of GPUs per node for training. Default is 0 (i.e. train on CPU)')
    parser.add_argument('--nodes', type=int, default=1, required=False,
            help='Specify the number of nodes used in training. Default is 0')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, required=True, 
            help='Specify the batch size per iteration of training')
    parser.add_argument('--num_workers', type=int, required=True,
            help='Specify the number of workers')
    parser.add_argument('--freeze_backbone', type=int, required=True,
            help='Specify whether to freeze layers EXCEPT the final layer for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=0.001, required=False)
    parser.add_argument('--epochs', type=int, default=10, required=False,
            help='Specify the total number of training epochs')
    
    # Dataset and sample configuration
    parser.add_argument('--sample_interval', type=int, default=10, 
            help='The interval between consecutive frames to sample. Default is 10')
    parser.add_argument('--seq_length', type=int, default=5,
            help='The length of the sequence to sample. Default is 5')
    parser.add_argument('--behaviour_threshold', type=int, default=72,
            help='The length of time (in frames) a behaviour must be exhibited to be a valid sample at training time. Default is 72')

    args = parser.parse_args()

    main(args)
