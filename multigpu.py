"""
MultiGPU

Workflow: 以手写数字体为例
1. 安装Pytorch Lightning
2. 基于pl构建dataloader
3. 构建用于训练的神经网络
4. 构建训练
"""

import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.functional as F
import torch.nn as nn


class MNISTDataModule(pl.LightningDataModule):
    """
    构建dataloader
    """
    def __init__(self, data_dir: str = "/home/kennys/workspace/MineX/data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform, download=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
    


class LitModel(pl.LightningModule):
    """
    构建用于训练的神经网络
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch # x为数据 y为标签
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def on_after_backward(self):
        # 记录第一层的梯度
        if self.trainer.global_step % 25 == 0: # 每25步记录一次
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.trainer.global_step)
    

if __name__ == "__main__":
    # 初始化数据模块和模型
    mnist_dm = MNISTDataModule()
    model = LitModel()

    # 初始化trainer
    trainer = Trainer(
        max_epochs=20,
        gpus=[0,1],
        progress_bar_refresh_rate=20,
        callbacks=[ModelCheckpoint(dirpath="checkpoints/", save_top_k=1, verbose=True, monitor="val_loss", mode="min")],
    )

    trainer.fit(model, mnist_dm)