import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import TWDataset
from twmodel import TWModel

class TWAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if self.cfg.use_gpu and torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = self.cfg.model_dir

        self.model = None

    def build_model(self):
        input_size = self.cfg.model.input_size
        hidden_size = self.cfg.model.hidden_size
        num_layers = self.cfg.model.num_layers
        output_size = self.cfg.model.output_size
        aval_type_size = self.cfg.model.aval_type_size
        self.model = TWModel(input_size, hidden_size, aval_type_size, num_layers, output_size)

    def build_trainval_data(self):
        self.train_dataset = TWDataset(data_dir=self.cfg.train_datadir , fnames=self.cfg.train_filenames)
        self.val_dataset = TWDataset(data_dir=self.cfg.val_datadir , fnames=self.cfg.var_filenames)

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=2)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=2)

    def train(self):
        self.best_loss = float("inf")

        self.build_model()
        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter(self.cfg.train_sum, "Train")
        self.val_writer = SummaryWriter(self.cfg.val_sum, "Val")

        for epoch in range(last_epoch + 1, epochs + 1):
            self.train_per_epoch(epoch)
            if epoch > 1:
                loss = self.validate_per_epoch(epoch)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.save_model(epoch)

        self.train_writer.close()
        self.val_writer.close()

    def train_per_epoch(self, epoch):
        total_loss = 0
        self.model.train()
        for batch_idx, (batch_fb, batch_doc, batch_occ, labels) in enumerate(self.train_data_loader):
            batch_fb, batch_doc, batch_occ = batch_fb.to(self.device), batch_doc.to(self.device),\
                                                        batch_occ.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_fb, batch_doc, batch_occ)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.train_writer.add_scalar('Loss', total_loss, epoch)
        print("epoch", epoch, "loss:", total_loss)

        return total_loss

    def validate_per_epoch(self, epoch):
        total_loss = 0
        self.model.eval()
        for batch_idx, (batch_fb, batch_doc, batch_occ, labels) in enumerate(self.val_data_loader):
            batch_fb, batch_doc, batch_occ = batch_fb.to(self.device), batch_doc.to(self.device),\
                                                        batch_occ.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_fb, batch_doc, batch_occ)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.val_writer.add_scalar('Loss', total_loss, epoch)
        print("epoch", epoch, "loss:", total_loss)

        return total_loss

    def make_prediction(self):
        pass

    def evaluate(self):
        pass

    def build_optimizer(self):
        optimizer = self.cfg.optimizer.used.lower()
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.Adam.lr)
        elif optimizer == "sgd":
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.SGD.lr)

    def build_loss_function(self):
        return nn.CrossEntropyLoss()

    def save_model(self, epoch):
        ckpt = {'my_classifier': self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.best_loss = ckpt['best_loss']

        return ckpt['epoch']
