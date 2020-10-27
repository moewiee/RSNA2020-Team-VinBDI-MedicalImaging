import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from cvcore.utils import save_checkpoint
import pandas as pd


def valid_model(_print, cfg, model, valid_loader,
                loss_function, score_function, epoch,
                best_metric=None, checkpoint=False):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, lb) in enumerate(tbar):
            image = image.cuda()
            if cfg.MODEL.NAME == "seriesnet":
                image = image.half()
                lb = lb.squeeze(1)
            else:
                lb = lb[1]
            if cfg.MODEL.NAME == "embeddingnet":
                _, _, second_w_output_1, second_w_output_2 = model(image)
                _, _, second_w_output_3, second_w_output_4 = model(image.flip(2))
                second_w_output = (second_w_output_1+second_w_output_2+second_w_output_3+second_w_output_4) / 4.
            elif cfg.MODEL.NAME == "seriesnet":
                w_output1, w_output2, w_output3 = model(image)
                second_w_output = (w_output1 + w_output2 + w_output3) / 6.
                w_output1, w_output2, w_output3 = model(image.flip(2))
                second_w_output += (w_output1 + w_output2 + w_output3) / 6.
            else:
                _, _, second_w_output = model(image)

            preds.append(second_w_output.cpu())
            if cfg.MODEL.NAME == "seriesnet":
                targets.append(lb.float())
            else:
                targets.append(lb.float().unsqueeze(-1))

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)



    # record
    val_loss = loss_function(preds.float(), targets)
    score = score_function(targets, torch.sigmoid(preds.float()) > 0.5, average='macro')

    _print(f"VAL LOSS: {val_loss:.5f}, SCORE: {score}")
    # checkpoint
    if checkpoint:
        is_best = val_loss < best_metric
        best_metric = min(val_loss, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.NAME,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric}
        save_filename = f"{cfg.NAME}.pth"
        if is_best: # only save best checkpoint, no need resume
            save_checkpoint(save_dict, is_best,
                            root=cfg.DIRS.WEIGHTS, filename=save_filename)
            print("score improved, saving new checkpoint...")
        return val_loss, best_metric


def test_model(_print, cfg, model, test_loader):
    # switch to evaluate mode
    model.eval()

    preds = []
    if cfg.MODEL.NAME == "embeddingnet":
        embeddings_pred = []
    names = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, name) in enumerate(tbar):
            image = image.cuda()
            if cfg.MODEL.NAME == "embeddingnet":
                w_output_1, w_output_2, second_w_output_1, second_w_output_2 = model(image)
                w_output_3, w_output_4, second_w_output_3, second_w_output_4 = model(image.flip(2))
                second_w_output = (second_w_output_1+second_w_output_2+second_w_output_3+second_w_output_4) / 4.
            elif cfg.MODEL.NAME == "seriesnet":
                image = image.half()
                w_output1, w_output2, w_output3 = model(image)
                second_w_output = (w_output1 + w_output2 + w_output3) / 6.
                w_output1, w_output2, w_output3 = model(image.flip(2))
                second_w_output += (w_output1 + w_output2 + w_output3) / 6.
            else:
                _, _, second_w_output = model(image)

            preds.append(torch.sigmoid(second_w_output.cpu().float()))
            if cfg.MODEL.NAME == "embeddingnet":
                embeddings_pred.append(torch.cat([w_output_1,second_w_output_1,w_output_2,second_w_output_2,w_output_3,second_w_output_3,w_output_4,second_w_output_4], dim=1))
            names.append(name)

    preds = torch.cat(preds, 0).numpy()
    if cfg.MODEL.NAME == "embeddingnet":
        embeddings_pred = torch.cat(embeddings_pred, 0)
    names = [n for name in names for n in name]
    if cfg.INFER.SAVE_NAME:
        for p,n in tqdm(zip(embeddings_pred, names)):
            np.save(os.path.join(cfg.DIRS.EMBEDDINGS, n), p.float().cpu().numpy())
        data = [[n,p[0]] for p,n in zip(preds, names)]
    else:
        if cfg.MODEL.NAME != "seriesnet":
            data = [[n,p[0]] for p,n in zip(preds, names)]
            pd.DataFrame(data=data, columns=['id', 'label']).to_csv("image_submission.csv", index=False)
        else:
            data = []
            for p,n in zip(preds, names):
                data.append([n+"_negative_exam_for_pe", p[0]])
                data.append([n+"_indeterminate", p[1]])
                data.append([n+"_chronic_pe", p[2]])
                data.append([n+"_acute_and_chronic_pe", p[3]])
                data.append([n+"_central_pe", p[4]])
                data.append([n+"_leftsided_pe", p[5]])
                data.append([n+"_rightsided_pe", p[6]])
                data.append([n+"_rv_lv_ratio_gte_1", p[7]])
                data.append([n+"_rv_lv_ratio_lt_1", p[8]])
            pd.DataFrame(data=data, columns=['id', 'label']).to_csv("exam_submission.csv", index=False)


def embedding_model(_print, cfg, model, valid_loader):
    # switch to evaluate mode
    model.eval()

    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for _, (image, name) in enumerate(tbar):
            image = image.cuda()
            _, embedding_output, _ = model(image)

            for e, n in zip(embedding_output, name):
                np.save(os.path.join(cfg.DIRS.EMBEDDINGS, n), e.float().cpu().numpy())
