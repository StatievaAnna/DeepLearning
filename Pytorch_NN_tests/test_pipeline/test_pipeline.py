import os
import sys
import random
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import wandb

PROJECT_ROOT = os.path.abspath(r"D:/загрузки/ml_mipt_dafe-/01_Pytorch_NN")
sys.path.insert(0, PROJECT_ROOT)

from example_project.hparams import config
from example_project.train import (
    compute_accuracy,
    get_transforms,
    load_datasets,
    create_data_loaders,
    initialize_model,
    create_optimizer,
    train_single_batch,
    evaluate_model,
    log_metrics,
    save_artifacts,
    train_model,
)

os.environ["WANDB_MODE"] = "online"

@pytest.fixture(scope="session", autouse=True)
def wandb_run():
    run = wandb.init(project="pytest_tests",
                 name="all_tests_run",
                 dir="wandb_logs")
    yield run
    wandb.finish()

@pytest.fixture
def tfm():
    return get_transforms()

@pytest.fixture
def datasets():
    train_ds, test_ds = load_datasets(root="CIFAR10", download=True)
    return train_ds, test_ds

@pytest.fixture
def small_loaders(datasets):
    train_ds, test_ds = datasets
    return create_data_loaders(train_ds, test_ds,
                               batch_size=config["batch_size"],
                               train_size=256, test_size=128)

CPU = torch.device("cpu")


def test_compute_accuracy_basic(wandb_run):
    preds   = torch.tensor([0, 1, 2, 2])
    targets = torch.tensor([0, 1, 1, 2])
    acc     = compute_accuracy(preds, targets)
    assert pytest.approx(acc.item()) == 0.75
    assert torch.allclose(acc, torch.tensor(0.75), atol=1e-5)

    wandb.log({"tests/test_compute_accuracy_basic": 1,
               "accuracy_sample": acc.item()})


def test_get_transforms_output_shape(wandb_run, tfm):
    img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    t   = tfm(img)
    assert t.shape == (3, 32, 32)
    assert t.dtype == torch.float32

    wandb.log({"tests/test_get_transforms_output_shape": 1})


def test_get_transforms_values(wandb_run, tfm, datasets):
    train_ds, _ = datasets
    idx = random.randint(0, len(train_ds) - 1)
    img_tensor, _ = train_ds[idx]
    img_pil = transforms.ToPILImage()(img_tensor)
    t = tfm(img_pil)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
    restored = (t * std) + mean
    assert torch.allclose(restored, transforms.ToTensor()(img_pil), atol=1e-5)

    wandb.log({"tests/test_get_transforms_values": 1})


def test_load_datasets_lengths(wandb_run, datasets):
    train_ds, test_ds = datasets
    assert len(train_ds) == 50_000
    assert len(test_ds) == 10_000
    wandb.log({"tests/test_load_datasets_lengths": 1})


def test_create_data_loaders_counts(wandb_run, datasets):
    train_ds, test_ds = datasets
    tr_loader, te_loader = create_data_loaders(train_ds, test_ds,
                                               batch_size=16, train_size=128, test_size=64)
    assert len(tr_loader.dataset) == 128
    assert len(te_loader.dataset) == 64
    wandb.log({"tests/test_create_data_loaders_counts": 1})


def test_initialize_model_forward(wandb_run):
    model = initialize_model(device=CPU, zero_init_residual=True)
    out   = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    wandb.log({"tests/test_initialize_model_forward": 1})


def test_create_optimizer_contains_params(wandb_run):
    lin = nn.Linear(10, 2)
    opt = create_optimizer(lin, lr=1e-3, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups[0]["params"]) > 0
    wandb.log({"tests/test_create_optimizer_contains_params": 1})


def test_train_single_batch_updates_weights(wandb_run, small_loaders):
    tr_loader, _ = small_loaders
    model = initialize_model(CPU, True)
    crit  = nn.CrossEntropyLoss()
    opt   = create_optimizer(model, lr=1e-3, weight_decay=0.0)
    batch = next(iter(tr_loader))
    before = [p.clone() for p in model.parameters() if p.requires_grad]
    loss   = train_single_batch(model, batch, crit, opt, CPU)
    assert loss.item() > 0
    assert any((not torch.equal(b, p) for b, p in zip(before, model.parameters()) if p.requires_grad))
    wandb.log({"tests/test_train_single_batch_updates_weights": 1,
               "loss_sample": loss.item()})


def test_evaluate_model_range(wandb_run, small_loaders):
    _, te_loader = small_loaders
    acc = evaluate_model(initialize_model(CPU, True), te_loader, CPU)
    assert 0.0 <= acc.item() <= 1.0
    wandb.log({"tests/test_evaluate_model_range": 1,
               "acc_sample": acc.item()})


def test_log_metrics_calls_wandb(wandb_run, monkeypatch):
    recorded = {}
    monkeypatch.setattr(wandb, "log",
                        lambda m, step=None: recorded.update({"metrics": m, "step": step}))
    log_metrics({"loss": 1.23}, epoch=0, batch_idx=0, batch_size=8, dataset_len=64)
    assert recorded["metrics"]["loss"] == 1.23 and recorded["step"] == 8
    wandb.log({"tests/test_log_metrics_calls_wandb": 1})


def test_save_artifacts(wandb_run, monkeypatch):
    monkeypatch.setattr(wandb, "run", type("Run", (), {"id": "TEST_RUN_ID"}))
    model = nn.Linear(10, 2)
    save_artifacts(model)
    assert Path("model.pt").exists()
    assert Path("run_id.txt").read_text().strip() == "TEST_RUN_ID"
    wandb.log({"tests/test_save_artifacts": 1})

# смок поверхностный тест

SMOKE_CFG = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "epochs": config["epochs"],
    "zero_init_residual": config["zero_init_residual"],
    "train_size": 128,
    "test_size": 64,
}

def test_train_model_smoke(wandb_run):
    run = wandb.init(project="pytest_tests",
                 name="all_tests_run",
                 dir="wandb_logs")
    train_model(SMOKE_CFG)
    assert Path("model.pt").exists() and Path("run_id.txt").exists()
    wandb.log({"tests/test_train_model_smoke": 1})
    wandb.finish()

HP_LR = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
HP_WD = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
HP_BS = [4, 8, 16, 32]
HP_CONFIGS = [
    {"lr": lr, "wd": wd, "bs": bs}
    for lr, wd, bs in itertools.product(HP_LR, HP_WD, HP_BS)
]

@pytest.mark.parametrize("hp", HP_CONFIGS)
def test_train_model_hyperparams(wandb_run, hp):
    cfg = dict(SMOKE_CFG, batch_size=hp["bs"],
               learning_rate=hp["lr"],
               weight_decay=hp["wd"],
               train_size=2000, test_size=1000)
    train_model(cfg)
    assert Path("model.pt").exists()
    wandb.log({f"tests/train_model_hyperparams": 1,
               "hp/lr": hp["lr"],
               "hp/wd": hp["wd"],
               "hp/bs": hp["bs"]})
