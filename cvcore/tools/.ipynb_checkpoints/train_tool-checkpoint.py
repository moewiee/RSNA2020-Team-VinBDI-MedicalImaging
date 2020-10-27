import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import os

from cvcore.data import cutmix_data, mixup_data
from cvcore.utils import AverageMeter, save_checkpoint
from cvcore.solver import WarmupCyclicalLR, WarmupMultiStepLR


def train_loop(_print, cfg, model, model_swa, train_loader,
               criterion, optimizer, scheduler, epoch, scaler):
    _print(f"\nEpoch {epoch + 1}")
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    for i, (image, target) in enumerate(tbar):
        image = image.cuda()
        if cfg.MODEL.NAME == "seriesnet":
            image = image.half()
            lb = target.cuda().squeeze(1)
        else:
            lb = target[0].cuda().long()
            second_lb = target[1].cuda().unsqueeze(-1).float()


        # mixup/ cutmix
        if cfg.DATA.MIXUP.ENABLED:
            image = mixup_data(image, alpha=cfg.DATA.MIXUP.ALPHA)
        elif cfg.DATA.CUTMIX.ENABLED:
            image = cutmix_data(image, alpha=cfg.DATA.CUTMIX.ALPHA)
        if cfg.MODEL.NAME == "embeddingnet":
            w_output1, w_output2, second_w_output1, second_w_output2 = model(image)
        elif cfg.MODEL.NAME == "seriesnet":
            w_output1, w_output2, w_output3 = model(image)
        else:
            _, w_output, second_w_output = model(image)
        with autocast():
            # compute loss
            if cfg.MODEL.NAME == "embeddingnet":
                loss = criterion(w_output1, lb) + criterion(w_output2, lb) + F.binary_cross_entropy_with_logits(second_w_output1, second_lb) + F.binary_cross_entropy_with_logits(second_w_output2, second_lb)
            elif cfg.MODEL.NAME == "seriesnet":
                loss = criterion(w_output1, lb) + criterion(w_output2, lb) + criterion(w_output3, lb)
            else:
                loss = criterion(w_output, lb) + F.binary_cross_entropy_with_logits(second_w_output, second_lb)
            # gradient accumulation
            loss = loss / cfg.SOLVER.GD_STEPS
        scaler.scale(loss).backward()
        # lr scheduler and optim. step
        if (i + 1) % cfg.SOLVER.GD_STEPS == 0:
            if isinstance(scheduler, WarmupCyclicalLR):
                scheduler(optimizer, i, epoch)
            elif isinstance(scheduler, WarmupMultiStepLR):
                scheduler.step()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        # record loss
        losses.update(loss.item() * cfg.SOLVER.GD_STEPS, lb.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (
            losses.avg, optimizer.param_groups[-1]['lr']))

        if model_swa is not None:
            if i % cfg.SOLVER.SWA.FREQ == 0:
                moving_average(model_swa, model, cfg.SOLVER.SWA.DECAY)

    if model_swa is not None:
        bn_update(train_loader, model_swa)

    _print("Train loss: %.5f, learning rate: %.6f" %
           (losses.avg, optimizer.param_groups[-1]['lr']))


def moving_average(net1, net2, decay):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data = param1.data * decay + param2.data * (1-decay)

def copy_model(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 0
        param1.data += param2.data

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    tbar = tqdm(loader)
    for i, (input, _) in enumerate(tbar):
#         input = input.cuda(non_blocking=True).half()
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))