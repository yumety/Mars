import copy
import torch
import torch.nn as nn
from pathlib import Path

class ModelEMA:
    
    def __init__(self, model, decay=0.9999, updates=0, device=None):
        self.ema = self.deepcopy(model)
        self.ema.eval()
        
        self.decay = decay
        self.updates = updates
        self.model = model
        self.device = device or next(model.parameters()).device
        
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    @staticmethod
    def deepcopy(model):
        if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module
        return copy.deepcopy(model).eval()
    
    def update(self, model=None):
        if model is not None:
            self.model = model
            
        self.updates += 1
        decay = self.decay
        
        if self.updates <= 1000:
            decay = min(decay, (1 + self.updates) / (10 + self.updates))
        
        model_state = self.model.state_dict()
        ema_state = self.ema.state_dict()
        
        with torch.no_grad():
            for k, v in ema_state.items():
                if v.dtype.is_floating_point:
                    v.mul_(decay)
                    v.add_(model_state[k].detach().to(self.device), alpha=1 - decay)
    
    def apply(self, model=None):
        target_model = model or self.model
        target_model.load_state_dict(self.state_dict())
        return target_model
    
    def state_dict(self):
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'updates': self.updates,
            'decay': self.decay
        }, Path(path))
    
    @classmethod
    def load(cls, path, model, device=None):
        checkpoint = torch.load(path, map_location=device)
        ema = cls(model, decay=checkpoint.get('decay', 0.9999), 
                 updates=checkpoint.get('updates', 0), device=device)
        ema.load_state_dict(checkpoint['state_dict'])
        return ema