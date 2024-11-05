import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils.conf import Configuration
from utils.sample import getSample, TSData
from model.TStransformer import TStransformer
from model.loss import ScaledL2Loss

import logging
import os
import json


class Experiment:
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.model_type = self.conf.getEntry("model_type")
        self.device = self.conf.getEntry("device")
        self.epoch_max = self.conf.getEntry("epoch_max")
        self.log_path = conf.getEntry("log_path")
        self.model_path = self.conf.getEntry("model_path")
        self.model_epoch_pos = self.conf.getEntry("model_epoch_pos")
        self.batch_size = self.conf.getEntry("batch_size")
        self.len_series = self.conf.getEntry("len_series")
        self.len_reduce = self.conf.getEntry("len_reduce")
        self.orth_regularizer = self.conf.getEntry("orth_regularizer")
        
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filename = self.log_path,
            filemode = "w"
        )
        
        with open(self.conf.getEntry("conf_path"), 'r') as infile:
            config_data = json.load(infile)
        logging.info("Configuration from example.json: %s", json.dumps(config_data, indent=4))
        logging.info(f"Experiment initialized with max epochs: {self.epoch_max} on device: {self.device}")
        
    
    def setup(self) -> None:
        train_samples, val_samples = getSample(self.conf)
        
        self.train_loader = DataLoader(TSData(train_samples), batch_size = self.batch_size, shuffle = True)
        self.train_query_loader = DataLoader(TSData(train_samples), batch_size = self.batch_size, shuffle = True)
        self.val_loader = DataLoader(TSData(val_samples), batch_size = self.batch_size, shuffle = True)
        self.val_query_loader = DataLoader(TSData(val_samples), batch_size = self.batch_size, shuffle = True)
        
        if self.model_type == "TStransformer":
            self.model = TStransformer(self.conf).to(self.device)
        elif self.model_type == "pretrained_TStransformer":
            self.model = PretrainedTStransformer(self.conf).to(self.device)
        else:
            raise NotImplementedError("Model type not implemented")
        
        if os.path.exists(self.conf.getEntry("model_path")):
            logging.info("Model loading...")
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            logging.info("Model initializing...")
            self.model = self.initModel(self.model)
        
        self.loss_calculator = ScaledL2Loss(self.len_series, self.len_reduce).to(self.device)
        
        self.optimizer = AdamW(self.model.parameters(), lr = 1e-4, weight_decay = 0.01)
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda = self.lr_lambda)
        
        self.orth_regularizer = self.conf.getEntry('orth_regularizer')
        if self.orth_regularizer == 'srip':
            self.srip_weight = self.conf.getEntry('srip_max')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if torch.cuda.device_count() > 1:
            selected_device = self.conf.getEntry("GPUs")
            logging.info(f"Using {len(selected_device)} GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=selected_device)
            
            
    def run(self) -> None:
        self.setup()
        
        self.epoch = 0
        while self.epoch < self.epoch_max:
            if self.orth_regularizer == 'srip':
                self.adjust_srip()
            
            self.epoch += 1
            
            self.train()
            self.validate()
            
            # Save the model every 5 epochs
            if self.epoch % 5 == 0:
                torch.save(self.model.state_dict(), f"{self.model_epoch_pos}epoch{self.epoch}.pth")
                logging.info(f"Model in epoch: {self.epoch} saved successfully.")

        torch.save(self.model.state_dict(), self.model_path)
        logging.info("Model saved successfully.")
            
    
    def train(self) -> None:
        logging.info(f'epoch: {self.epoch}, start training')
        
        for one_batch, another_batch in zip(self.train_loader, self.train_query_loader):
            self.optimizer.zero_grad()
            
            one_batch = one_batch.to(self.device)   # (batch_size, dim_series, len_series)
            one_batch_reduce = self.model(one_batch)    # (batch_size, dim_series, len_reduce)
            
            with torch.no_grad():
                another_batch = another_batch.to(self.device)
                another_batch_reduce = self.model(another_batch)
                
            loss = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    
    def validate(self) -> None:
        errors = []
        
        with torch.no_grad():
            for one_batch, another_batch in zip(self.val_loader, self.val_query_loader):
                one_batch = one_batch.to(self.device)
                another_batch = another_batch.to(self.device)
                one_batch_reduce = self.model(one_batch)
                another_batch_reduce = self.model(another_batch)
                
                err = self.loss_calculator(one_batch, another_batch, one_batch_reduce, another_batch_reduce)
                errors.append(err.cpu())
                
                # logging.info(
                #     f"one_batch: {one_batch.cpu().numpy()}\n"
                #     f"another_batch: {another_batch.cpu().numpy()}\n"
                #     f"one_batch_reduce: {one_batch_reduce.cpu().numpy()}\n"
                #     f"another_batch_reduce: {another_batch_reduce.cpu().numpy()}\n"
                #     f"err: {err.cpu().item()}\n\n"
                # )
                
        avg_error = torch.mean(torch.stack(errors)).item()
        logging.info(f'epoch: {self.epoch}, validate trans_err: {avg_error:.10f}')
    
    
    def lr_lambda(self, step):
            warmup_steps = 4000
            return min((step + 1) / warmup_steps, 1.0 / (step + 1) ** 0.5)
            
        
    def initModel(self, model):
        def initialize_weights(module):
            if isinstance(module, nn.Linear):
                # 对于线性层，使用 Xavier 均匀分布初始化权重
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 对于嵌入层，使用 Xavier 均匀分布初始化
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                # 对于 LayerNorm，初始化偏置和权重
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # 使用 apply 方法将初始化函数应用于模型中的每个模块
        model.apply(initialize_weights)
        
        return model
    

    def adjust_srip(self):
        if self.conf.getEntry('srip_mode') == 'linear':
            srip_max = self.conf.getEntry('srip_max')
            srip_min = self.conf.getEntry('srip_min')
            self.srip_weight = srip_max - self.epoch * (srip_max - srip_min) / self.epoch_max
            
