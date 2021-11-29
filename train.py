import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorchvideo.models.resnet
from pytorchvideo.models.head import create_res_basic_head
from dataset.datamodule import PanAfDataModule

class VideoClassificationLightningModule(pl.LightningModule):
  
    def __init__(self, model_name, freeze_backbone):
      super().__init__()


      self.model_name = model_name
      self.freeze_backbone = freeze_backbone
    
      # Load pretrained model
      pretrained_model = torch.hub.load("facebookresearch/pytorchvideo:main", model=self.model_name, pretrained=True)
      
      # Strip the head from backbone  
      self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

      # Attach a new head with specified class number (hard coded for now...)
      self.head = create_res_basic_head(
              in_features=2048, out_features=9
      )
      
      if self.freeze_backbone:
          for param in self.backbone.parameters():
              param.requires_grad = False
      
      # Metric initialisation
      self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
      self.top3_train_accuracy = torchmetrics.Accuracy(top_k=3)
      self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)  
      self.top3_val_accuracy = torchmetrics.Accuracy(top_k=3) 

    def forward(self, x):
      return self.head(self.backbone(x))

    def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the
      # format provided by the dataset
      data, label, meta = batch
      pred = self(data)

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(pred, label)
      
      top1_train_acc = self.top1_train_accuracy(pred, label)
      top3_train_acc = self.top3_train_accuracy(pred, label)
      
      self.log('top1_train_acc', top1_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=True)
      self.log('top3_train_acc', top3_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=True)
      self.log('train_loss', loss.item(), logger=True, on_epoch=True, on_step=True)

      return {"loss": loss, "logs": {"train_loss": loss, "top1_train_acc": top1_train_acc, "top3_train_acc": top3_train_acc}}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_train_accuracy.compute()
        top3_acc = self.top3_train_accuracy.compute()
        self.log('train_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)
        self.log('train_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)  

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
 

    def validation_step(self, batch, batch_idx):
  
      data, label, meta = batch
      pred = self(data)
      loss = F.cross_entropy(pred, label)
      self.log("val_loss", loss)
      
      top1_val_acc = self.top1_val_accuracy(pred, label)
      top3_val_acc = self.top3_val_accuracy(pred, label)
      
      self.log('top1_train_acc', top1_val_acc, logger=False, on_epoch=False, on_step=False, prog_bar=True)
      self.log('top3_train_acc', top3_val_acc, logger=False, on_epoch=False, on_step=False, prog_bar=True)
      self.log('train_loss', loss, logger=True, on_epoch=True, on_step=True)
      
      return {"loss": loss, "logs": {"val_loss": loss, "top1_val_acc": top1_val_acc, "top3_val_acc": top3_val_acc}}

    def validation_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_val_accuracy.compute()
        top3_acc = self.top3_val_accuracy.compute()
        self.log('val_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)  

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
 
    def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=1e-1)

def main():
    
    # Input all needs to come for argparse eventually...
    classification_module = VideoClassificationLightningModule(model_name='slow_r50', freeze_backbone=True)
    data_module = PanAfDataModule()
    trainer = pl.Trainer()
    trainer.fit(classification_module, data_module)

if __name__== "__main__":
    main()
