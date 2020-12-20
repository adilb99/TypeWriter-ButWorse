import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import TWDataset, collate_fn
from twmodel import TWModel

class TWAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if self.cfg.use_gpu and torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = self.cfg.model_dir
        print (self.device)
        if self.cfg.use_gpu and torch.cuda.is_available():
            print (torch.cuda.get_device_name(torch.cuda.current_device()))
        self.model = None

    def build_model(self):
        input_size = self.cfg.model.input_size
        hidden_size = self.cfg.model.hidden_size
        num_layers = self.cfg.model.num_layers
        output_size = self.cfg.model.output_size
        self.model = TWModel(input_size, hidden_size, num_layers, output_size).to(self.device)

    def build_trainval_data(self):
        self.train_dataset = TWDataset(data_dir=self.cfg.train_datadir , fnames=self.cfg.train_filenames)
        self.val_dataset = TWDataset(data_dir=self.cfg.val_datadir , fnames=self.cfg.var_filenames)

        self.train_steps = len(self.train_dataset) // self.cfg.batch_size
        self.val_steps = len(self.val_dataset) // self.cfg.batch_size

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,\
                                            collate_fn=collate_fn,shuffle=True, num_workers=2)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,\
                                            collate_fn=collate_fn,shuffle=False, num_workers=2)

    def train(self):
        self.best_loss = float("inf")
        self.best_acc = 0
        self.f1 = 0

        self.build_model()
        self.criterion = self.build_loss_function()
        self.optimizer = self.build_optimizer()

        self.build_trainval_data()

        last_epoch = 0
        if os.path.exists(self.checkpoint_dir):
            last_epoch = self.load_model()

        epochs = self.cfg.epochs
        self.train_writer = SummaryWriter(self.cfg.train_sum, "Train")
        self.val_writer = SummaryWriter(self.cfg.val_sum, "Val")

        for epoch in range(last_epoch + 1, epochs + 1):
            self.train_per_epoch(epoch)
            if epoch > 1:
                loss, acc, f1 = self.validate_per_epoch(epoch)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_acc = acc
                    self.f1 = f1
                    self.save_model(epoch)

        self.train_writer.close()
        self.val_writer.close()

    def train_per_epoch(self, epoch):
        total_acc = 0
        tqdm_batch = tqdm(total=self.train_steps, dynamic_ncols=True)
        total_loss = 0

        all_labels = []
        all_outs = []

        self.model.train()
        for batch_idx, (batch_fb, batch_doc, batch_occ, labels) in enumerate(self.train_data_loader):
            batch_fb, batch_occ = batch_fb.to(torch.float32), batch_occ.to(torch.float32)
            labels = labels.to(torch.float32)
            labels = torch.argmax(labels, dim=1)
            batch_fb, batch_doc, batch_occ = batch_fb.to(self.device), batch_doc.to(self.device),\
                                                        batch_occ.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_fb, batch_doc, batch_occ)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            outputs = outputs.argmax(dim=1)
            acc = (outputs == labels).float().sum()
            total_acc += acc

            all_labels.append(labels.cpu().numpy())
            all_outs.append(outputs.cpu().numpy())

            tqdm_update = "Epoch={0:04d},loss={1:.4f}, acc={2:.4f}".format(epoch, loss.item(), acc / labels.shape[0])
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_acc /= len(self.train_dataset)
        all_labels = np.concatenate(all_labels)
        all_outs = np.concatenate(all_outs)
        f1 = f1_score(all_labels, all_outs, average='weighted')

        self.train_writer.add_scalar('Loss', total_loss, epoch)
        self.train_writer.add_scalar('acc', total_acc, epoch)
        self.train_writer.add_scalar('f1', f1, epoch)
        print("epoch", epoch, "loss:", total_loss, "accuracy", total_acc, "f1", f1)

        tqdm_batch.close()
        return total_loss, total_acc, f1

    def validate_per_epoch(self, epoch):
        total_acc = 0
        tqdm_batch = tqdm(total=self.val_steps, dynamic_ncols=True)
        total_loss = 0

        all_labels = []
        all_outs = []

        self.model.eval()
        for batch_idx, (batch_fb, batch_doc, batch_occ, labels) in enumerate(self.val_data_loader):
            batch_fb, batch_occ = batch_fb.to(torch.float32), batch_occ.to(torch.float32)
            labels = labels.to(torch.float32)
            labels = torch.argmax(labels, dim=1)
            batch_fb, batch_doc, batch_occ = batch_fb.to(self.device), batch_doc.to(self.device),\
                                                        batch_occ.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(batch_fb, batch_doc, batch_occ)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            outputs = outputs.argmax(dim=1)
            acc = (outputs == labels).float().sum()
            total_acc += acc

            all_labels.append(labels.cpu().numpy())
            all_outs.append(outputs.cpu().numpy())

            tqdm_update = "Epoch={0:04d},loss={1:.4f}, acc={2:.4f}".format(epoch, loss.item(), acc / labels.shape[0])
            tqdm_batch.set_postfix_str(tqdm_update)
            tqdm_batch.update()

        total_acc /= len(self.val_dataset)
        all_labels = np.concatenate(all_labels)
        all_outs = np.concatenate(all_outs)
        f1 = f1_score(all_labels, all_outs, average='weighted')

        self.val_writer.add_scalar('Loss', total_loss, epoch)
        self.val_writer.add_scalar('acc', total_acc, epoch)
        self.val_writer.add_scalar('f1', f1, epoch)
        print("epoch", epoch, "loss:", total_loss, "accuracy", total_acc, "f1", f1)

        tqdm_batch.close()
        return total_loss, total_acc, f1

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
                "best_acc": self.best_acc,
                "f1": self.f1,
                "epoch": epoch}
        torch.save(ckpt, self.checkpoint_dir)

    def load_model(self):
        ckpt = torch.load(self.checkpoint_dir)
        self.model.load_state_dict(ckpt['model']).to(self.device)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.best_loss = ckpt['best_loss']
        self.best_acc = ckpt['best_acc']
        self.f1 = ckpt["f1"]

        return ckpt['epoch']
