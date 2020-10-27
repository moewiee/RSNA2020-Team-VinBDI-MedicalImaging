import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from torch.nn.functional import dropout
import random


class NormSoftmax(nn.Module):
    def __init__(self, in_features, out_features, temperature=1.):
        super(NormSoftmax, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data)

        self.ln = nn.LayerNorm(in_features, elementwise_affine=False)
        self.temperature = nn.Parameter(torch.Tensor([temperature]))

    def forward(self, x):
        x = self.ln(x)
        x = torch.matmul(F.normalize(x), F.normalize(self.weight))
        x = x / self.temperature
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet B0-B8.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()
        self.cfg = cfg

        backbone = timm.create_model(
            model_name=self.cfg.MODEL.NAME,
            pretrained=self.cfg.MODEL.PRETRAINED,
            in_chans=self.cfg.DATA.INP_CHANNEL,
            drop_path_rate=self.cfg.MODEL.DROPPATH,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        ### Original blocks ###
        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2
        if self.cfg.MODEL.POOL == "adaptive_pooling":
            self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.num_features = backbone.num_features
        if self.cfg.MODEL.HYPER:
            self.num_features = backbone.num_features + self.block4[-1].bn3.num_features + \
                                self.block5[-1].bn3.num_features
        ### Baseline head ###
        if self.cfg.MODEL.CLS_HEAD == "linear":
            self.fc = nn.Linear(self.num_features, self.cfg.MODEL.NUM_CLASSES)
        elif self.cfg.MODEL.CLS_HEAD == "norm":
            self.fc = NormSoftmax(self.num_features, self.cfg.MODEL.NUM_CLASSES)
        self.second_fc = nn.Linear(self.cfg.MODEL.NUM_CLASSES, 1)
        del backbone

    def _features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x); b4 = x
        x = self.block5(x); b5 = x
        x = self.block6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return b4,b5,x

    def forward(self, x):
        with autocast():
            b4, b5, x = self._features(x)
            x = self.global_pool(x)
            if self.cfg.MODEL.HYPER:
                b4 = self.global_pool(b4)
                b5 = self.global_pool(b5)
                x = torch.cat([x, b4, b5], 1)
            x = torch.flatten(x, 1)
            embeddings = x
            if self.cfg.MODEL.DROPOUT > 0.:
                x = torch.nn.functional.dropout(x, self.cfg.MODEL.DROPOUT, training=self.training)
            logits = self.fc(x)
            second_logits = self.second_fc(logits)
            return embeddings, logits, second_logits


class EmbeddingNet(nn.Module):
    def __init__(self, cfg):
        super(EmbeddingNet, self).__init__()        
        self.fw = nn.Sequential(
            nn.Conv2d(1, 32, 7, 1, 3),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            SelectAdaptivePool2d(pool_type="avg"),
            nn.Flatten(),
            nn.Linear(256, 7)
        )
        self.conv_out = nn.Linear(7, 1)

        self.lstm1 = nn.LSTM(7, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(128 * 2, 128, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(128*2, 128*2)
        self.linear2 = nn.Linear(128*2, 128*2)
        self.linear = nn.Linear(128*2, 7)
        self.lstm_out = nn.Linear(7, 1)

    def forward(self, x):
        with autocast():
            embedding_vector = x.squeeze(1)
            logits1 = self.fw(x)
            second_logits1 = self.conv_out((logits1))

            h_lstm1, _ = self.lstm1(embedding_vector)
            h_lstm2, _ = self.lstm2(h_lstm1)
            h_conc_linear1  = F.relu(self.linear1(h_lstm1))
            h_conc_linear2  = F.relu(self.linear2(h_lstm2))
            hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2
            logits2 = self.linear(hidden)[:,-1,:]
            second_logits2 = self.lstm_out((logits2))

            return logits1, logits2, second_logits1, second_logits2


class SeriesEmbeddingNet(nn.Module):
    def __init__(self, cfg):
        super(SeriesEmbeddingNet, self).__init__()
        
        self.netA = timm.create_model(
            model_name="tf_efficientnet_b0",
            pretrained=False,
            in_chans=1
        )
        self.netA.classifier = NormSoftmax(self.netA.num_features, 9)
        
        self.netB = timm.create_model(
            model_name="tf_efficientnet_b0",
            pretrained=False,
            in_chans=1
        )
        self.netB.classifier = NormSoftmax(self.netB.num_features, 9)
        
        self.netC = timm.create_model(
            model_name="tf_efficientnet_b0",
            pretrained=False,
            in_chans=1
        )
        self.netC.classifier = NormSoftmax(self.netC.num_features, 9)
    
    def forward(self, x):
        with autocast():
            if self.training:
                x1 = x if random.random() < 0.5 else x.flip(2)
                x2 = x if random.random() < 0.5 else x.flip(2)
                x3 = x if random.random() < 0.5 else x.flip(2)
            else:
                x1=x2=x3=x
            logits1 = self.netA(x1)
            logits2 = self.netB(x2)
            logits3 = self.netC(x3)
            return logits1, logits2, logits3


def build_model(cfg):
    model = None
    if "efficientnet" in cfg.MODEL.NAME:
        model = EfficientNet
    elif "embeddingnet" == cfg.MODEL.NAME:
        model = EmbeddingNet
    elif "seriesnet" == cfg.MODEL.NAME:
        model = SeriesEmbeddingNet
    return model(cfg)
