import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import TWModel

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if self.cfg.use_gpu and torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = self.cfg.model_dir

        self.model = None

    def build_model(self, ):
        input_size = self.cfg.model.input_size
        hidden_size = self.cfg.model.hidden_size
        num_layers = self.cfg.model.num_layers
        output_size = self.cfg.model.output_size
        self.model = TWModel(input_size, hidden_size, num_layers, output_size)
    
    def build_trainval_data(self):
        self.train_data_loader = None
        self.val_data_loader = None
        # TODO: build data loaders

    def train(self):
        self.best_loss = float("inf")

        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter("path", "Train")
        self.val_writer = SummaryWriter("path", "Val")

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
        for batch_idx, (batch_id, batch_tok, batch_cm, batch_type, labels) in enumerate(self.train_data_loader):
            batch_id, batch_tok, batch_cm, batch_type = batch_id.to(self.device), batch_tok.to(self.device),\
                                                        batch_cm.to(self.device), batch_type.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_id, batch_tok, batch_cm, batch_type)
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
        for batch_idx, (batch_id, batch_tok, batch_cm, batch_type, labels) in enumerate(self.val_data_loader):
            batch_id, batch_tok, batch_cm, batch_type = batch_id.to(self.device), batch_tok.to(self.device),\
                                                        batch_cm.to(self.device), batch_type.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_id, batch_tok, batch_cm, batch_type)
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
