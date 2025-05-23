import torch
from torch import nn
import torch.nn.functional as F
from overrides import override # this could be removed since Python 3.12

class CWDLoss(nn.Module):
    def __init__(self, device, temp=4.0):
        super().__init__()
        self.temp = temp
        self.device = device

    def forward(self, s_feats, t_feats):
        total_loss = 0.0
        for s, t in zip(s_feats, t_feats):
            if s.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(t, size=s.shape[-2:], mode='bilinear', align_corners=False)
            
            s_norm = F.log_softmax(s.flatten(2) / self.temp, dim=-1)
            t_norm = F.softmax(t.flatten(2).detach() / self.temp, dim=-1)
            
            loss = F.kl_div(s_norm, t_norm, reduction='batchmean') * (self.temp ** 2)
            total_loss += loss
        return total_loss / len(s_feats)
    
    
class ResponseLoss(nn.Module):
    def __init__(self, device, num_classes, teacher_class_idx, temp=4.0):
        super().__init__()
        self.temp = temp
        self.device = device
        self.num_classes = num_classes
        self.teacher_class_idx = teacher_class_idx

    def forward(self, s_preds, t_preds):
        total_loss = 0.0
        for s, t in zip(s_preds, t_preds):
            if s.shape[1] != t.shape[1]:
                t = t[:, self.teacher_class_idx, :, :]
            
            s_prob = F.log_softmax(s / self.temp, dim=1)
            t_prob = F.softmax(t.detach() / self.temp, dim=1)
            
            loss = F.kl_div(s_prob, t_prob, reduction='batchmean') * (self.temp ** 2)
            total_loss += loss
        return total_loss / len(s_preds)
    

class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        from train.loss import DetectionLoss
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CWDLoss(device=self.mcfg.device,temp=self.mcfg.distil_temperature)
        self.respLoss = ResponseLoss(self.mcfg.device, self.mcfg.nc, self.mcfg.teacherClassIndexes,self.mcfg.distil_temperature)
        # raise NotImplementedError("DistillationDetectionLoss::__init__")

    @override
    def __call__(self, rawPreds, batch):

        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()
