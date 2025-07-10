import yaml
import numpy as np
import torch
import torchvision
import torch.nn as nn
import random

from train_process import BaseTrainProcess
from classifier import Classifier, init_dataC
from load_data import init_data
from simclr import PreModel

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_simclr():
    with open('hyp_params.yaml', 'r') as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)
    set_seed(hyps['seed'])

    train_loader, valid_loader, _, _ = init_data(hyps)

    simclr_model = BaseTrainProcess(hyps, train_loader, valid_loader)

    train_losses, valid_losses = simclr_model.run()
    return simclr_model, train_losses, valid_losses

def train_no_simclr_classifier(num_epoch=30):
    with open('hyp_params.yaml', 'r') as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)
    set_seed(hyps['seed'])
    resnet50 = torchvision.models.resnet50(pretrained=True)
    encoder = nn.Sequential(*list(resnet50.children())[:-1])

    train_loader, valid_loader = init_dataC(hyps)
    classifier = Classifier(encoder)
    classifier.run(train_loader, valid_loader, num_epoch)

def train_simclr_classifier(num_epoch=30):
    with open('hyp_params.yaml', 'r') as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)
    set_seed(hyps['seed'])
    checkpoint = torch.load('best.pt')
    model = PreModel(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    resnet50_trained = model.encoder
    train_loader, valid_loader = init_dataC(hyps)
    classifier_pretrain = Classifier(resnet50_trained)

    classifier_pretrain.run(train_loader, valid_loader, num_epoch)

if __name__== "__main__":
    # train_simclr()
    # train_no_simclr_classifier(5)
    train_simclr_classifier(5)
