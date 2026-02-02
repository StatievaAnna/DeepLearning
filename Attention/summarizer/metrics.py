import torch
import torch.nn as nn
import wandb

DEVICE = torch.device('cuda')

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, padding_idx=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx  # индекс паддинг токена

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = -torch.sum(true_dist * pred, dim=self.dim)

        if self.padding_idx is not None:
            mask = (target != self.padding_idx)
            mask = torch.tensor(mask).to(DEVICE)
            loss = loss.masked_select(mask)

        return loss.mean()
    
from rouge import Rouge
rouge = Rouge()

def evaluate_rouge(predictions, references):
    scores = rouge.get_scores(predictions, references, avg=True)
    # for k, v in scores.items():
    #     print(f"{k}: {v}")
    return scores

def log_rouge_to_wandb(scores):
    wandb.log({
        f"{k}/{m}": v[m]
        for k, v in scores.items()
        for m in v
    })