# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import pydicom
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.functional import dropout
import random
import timm
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from collections import OrderedDict
from tqdm import tqdm

def convert(dicom_path):
    series = os.listdir(dicom_path)[0]
    instance = os.listdir(os.path.join(dicom_path, series))
    stack_image = np.zeros(shape=(len(instance),512,512,3), dtype=np.uint8)
    stack_zpos = np.zeros(shape=(len(instance),))
    for i, ins in enumerate(instance):
        f = pydicom.dcmread(os.path.join(dicom_path, series, ins))
        image = f.pixel_array
        image = image.astype(np.int16)
        image[image <= -1000] = 0
        _, _, intercept, slope = get_windowing(f)
        image = image * slope + intercept
        image_c1 = window(image, -600, 1500)
        image_c2 = window(image, 100, 700)
        image_c3 = window(image, 40, 400)
        image = np.stack([image_c1, image_c2, image_c3], axis=-1)
        image = image[:,:,::-1]
        stack_image[i,...] = image
        z_pos = f.ImagePositionPatient[-1]
        stack_zpos[i] = z_pos
    stack_image = stack_image[np.argsort(stack_zpos)]
    instance = np.array(instance)[np.argsort(stack_zpos)]
    
    return stack_image, instance

def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue: return int(x[0])
    else: return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

class to_tensor_albu:
    def __init__(self):
        transformation = [Normalize(),
                           ToTensor()]
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class SeriesDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.to_tensor = to_tensor_albu()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.to_tensor(image)
        return image

class CatEmbeddingDataset(Dataset):
    def __init__(self, images):
        self.input = images
        self.sl = 31
        self.images = []
        for i in range(len(images) - self.sl + 1):
            self.images.append(images[i:i+31].unsqueeze(0))
        self.images = torch.cat(self.images)

        
    def __len__(self):
        return len(self.images - 30)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image        

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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
    def __init__(self, name):
        super(EfficientNet, self).__init__()

        backbone = timm.create_model(
            model_name=name,
            pretrained=False,
            in_chans=3,
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
        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.num_features = backbone.num_features

        ### Baseline head ###
        self.fc = nn.Linear(self.num_features, 7)
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
            x = torch.flatten(x, 1)
            logits = self.fc(x)
            return logits


class EmbeddingNet(nn.Module):
    def __init__(self):
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
            embedding_vector = x.squeeze(1).float()
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
    def __init__(self):
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
            logits1 = self.netA(x)
            logits2 = self.netB(x)
            logits3 = self.netC(x)
            return (logits1 + logits2 + logits3) / 3.

backbone_b4f0 = EfficientNet("tf_efficientnet_b4").cuda()
checkpoint = torch.load("weights/b4_sz512_fold0.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
    if "second_" not in k:
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
backbone_b4f0.load_state_dict(checkpoint_surged)
backbone_b4f0.eval()
del checkpoint_surged
backbone_b4f1 = EfficientNet("tf_efficientnet_b5").cuda()
checkpoint = torch.load("weights/b5_sz512_fold0.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
    if "second_" not in k:
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
backbone_b4f1.load_state_dict(checkpoint_surged)
backbone_b4f1.eval()
del checkpoint_surged
backbone_b4f2 = EfficientNet("tf_efficientnet_b3").cuda()
checkpoint = torch.load("weights/b3_sz512_fold1.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
    if "second_" not in k:
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
backbone_b4f2.load_state_dict(checkpoint_surged)
backbone_b4f2.eval()
del checkpoint_surged

embeddingnet_b4f0 = EmbeddingNet().cuda()
checkpoint = torch.load("weights/cnn_lstm_embeddings_b4_sz512_fold0.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
embeddingnet_b4f0.load_state_dict(checkpoint_surged)
embeddingnet_b4f0.eval()
del checkpoint_surged
embeddingnet_b4f1 = EmbeddingNet().cuda()
checkpoint = torch.load("weights/cnn_lstm_embeddings_b5_sz512_fold0.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
embeddingnet_b4f1.load_state_dict(checkpoint_surged)
embeddingnet_b4f1.eval()
del checkpoint_surged
embeddingnet_b4f2 = EmbeddingNet().cuda()
checkpoint = torch.load("weights/cnn_lstm_embeddings_b3_sz512_fold1.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
embeddingnet_b4f2.load_state_dict(checkpoint_surged)
embeddingnet_b4f2.eval()
del checkpoint_surged

seriesnet_b4f0 = SeriesEmbeddingNet().cuda()
checkpoint = torch.load("weights/triple_b0_series_b4_sz512_fold0.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
seriesnet_b4f0.load_state_dict(checkpoint_surged)
seriesnet_b4f0.eval()
del checkpoint_surged
seriesnet_b4f1 = SeriesEmbeddingNet().cuda()
checkpoint = torch.load("weights/triple_b0_series_b5_sz512_fold0.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
seriesnet_b4f1.load_state_dict(checkpoint_surged)
seriesnet_b4f1.eval()
del checkpoint_surged
seriesnet_b4f2 = SeriesEmbeddingNet().cuda()
checkpoint = torch.load("weights/triple_b0_series_b3_sz512_fold1.pth", "cpu").pop('state_dict')
checkpoint_surged = OrderedDict()
for k, v in checkpoint.items():
        checkpoint_surged[k.replace("module.","")] = v
del checkpoint
seriesnet_b4f2.load_state_dict(checkpoint_surged)
seriesnet_b4f2.eval()
del checkpoint_surged

BATCH_SIZE = 32
test_df=pd.read_csv('/media/datnt/data/kaggle-ct-test/test.csv')
submission_csv_data = []
for study in tqdm(test_df["StudyInstanceUID"].unique()):
    stack_image, instance_list = convert(f"/media/datnt/data/kaggle-ct-test/{study}")
    stack_image = SeriesDataset(stack_image)


    f0_embeddings_out = []
    f1_embeddings_out = []
    f2_embeddings_out = []
    for images in batch(stack_image, BATCH_SIZE):
        images = images.permute(1,0,2,3).cuda().half()
        with autocast():
            with torch.no_grad():
                f0_embeddings_out.append(backbone_b4f0(images))
                f1_embeddings_out.append(backbone_b4f1(images))
                f2_embeddings_out.append(backbone_b4f2(images))
    f0_embeddings_out = torch.cat(f0_embeddings_out)
    f1_embeddings_out = torch.cat(f1_embeddings_out)
    f2_embeddings_out = torch.cat(f2_embeddings_out)

    f0_first_slices = torch.cat([f0_embeddings_out[0,:].unsqueeze(0) for _ in range(15)])
    f0_last_slices = torch.cat([f0_embeddings_out[-1,:].unsqueeze(0) for _ in range(15)])
    f0_embeddings_out = torch.cat([f0_first_slices, f0_embeddings_out, f0_last_slices])
    f0_embeddings_out = CatEmbeddingDataset(f0_embeddings_out)
    f1_first_slices = torch.cat([f1_embeddings_out[0,:].unsqueeze(0) for _ in range(15)])
    f1_last_slices = torch.cat([f1_embeddings_out[-1,:].unsqueeze(0) for _ in range(15)])
    f1_embeddings_out = torch.cat([f1_first_slices, f1_embeddings_out, f1_last_slices])
    f1_embeddings_out = CatEmbeddingDataset(f1_embeddings_out)
    f2_first_slices = torch.cat([f2_embeddings_out[0,:].unsqueeze(0) for _ in range(15)])
    f2_last_slices = torch.cat([f2_embeddings_out[-1,:].unsqueeze(0) for _ in range(15)])
    f2_embeddings_out = torch.cat([f2_first_slices, f2_embeddings_out, f2_last_slices])
    f2_embeddings_out = CatEmbeddingDataset(f2_embeddings_out)
    
    f0_embeddings_stage2 = []
    f1_embeddings_stage2 = []
    f2_embeddings_stage2 = []
    images_output = []
    for images_f0, images_f1, images_f2 in zip(batch(f0_embeddings_out, BATCH_SIZE), batch(f1_embeddings_out, BATCH_SIZE), batch(f2_embeddings_out, BATCH_SIZE)):
        images_f0 = images_f0.unsqueeze(1).cuda()
        images_f1 = images_f1.unsqueeze(1).cuda()
        images_f2 = images_f2.unsqueeze(1).cuda()
        with autocast():
            with torch.no_grad():
                w_output_1_f0, w_output_2_f0, second_w_output_1_f0, second_w_output_2_f0 = embeddingnet_b4f0(images_f0)
                w_output_3_f0, w_output_4_f0, second_w_output_3_f0, second_w_output_4_f0 = embeddingnet_b4f0(images_f0.flip(2))
                w_output_1_f1, w_output_2_f1, second_w_output_1_f1, second_w_output_2_f1 = embeddingnet_b4f1(images_f1)
                w_output_3_f1, w_output_4_f1, second_w_output_3_f1, second_w_output_4_f1 = embeddingnet_b4f1(images_f1.flip(2))
                w_output_1_f2, w_output_2_f2, second_w_output_1_f2, second_w_output_2_f2 = embeddingnet_b4f2(images_f2)
                w_output_3_f2, w_output_4_f2, second_w_output_3_f2, second_w_output_4_f2 = embeddingnet_b4f2(images_f2.flip(2))
                ensemble_out = second_w_output_1_f0+second_w_output_2_f0+second_w_output_3_f0+second_w_output_4_f0
                ensemble_out += second_w_output_1_f1+second_w_output_2_f1+second_w_output_3_f1+second_w_output_4_f1
                ensemble_out += second_w_output_1_f2+second_w_output_2_f2+second_w_output_3_f2+second_w_output_4_f2 
                ensemble_out = torch.sigmoid(ensemble_out / 12.)
                images_output.append(ensemble_out)
                f0_embeddings_stage2.append(torch.cat([w_output_1_f0,second_w_output_1_f0,w_output_2_f0,second_w_output_2_f0,w_output_3_f0,second_w_output_3_f0,w_output_4_f0,second_w_output_4_f0], dim=1))
                f1_embeddings_stage2.append(torch.cat([w_output_1_f1,second_w_output_1_f1,w_output_2_f1,second_w_output_2_f1,w_output_3_f1,second_w_output_3_f1,w_output_4_f1,second_w_output_4_f1], dim=1))
                f2_embeddings_stage2.append(torch.cat([w_output_1_f2,second_w_output_1_f2,w_output_2_f2,second_w_output_2_f2,w_output_3_f2,second_w_output_3_f2,w_output_4_f2,second_w_output_4_f2], dim=1))
    images_output = torch.cat(images_output).cpu().numpy()
    for instanceuid,image_out in zip(instance_list, images_output):
        submission_csv_data.append([instanceuid.replace(".dcm",""), image_out.reshape(-1)[0]])

    f0_embeddings_stage2 = torch.cat(f0_embeddings_stage2, dim=0)
    f1_embeddings_stage2 = torch.cat(f1_embeddings_stage2, dim=0)
    f2_embeddings_stage2 = torch.cat(f2_embeddings_stage2, dim=0)


    seq_len = f0_embeddings_stage2.size()[0]
    padsize = 1024 - seq_len
    before_pad = int(padsize / 2)
    after_pad = int(padsize / 2)
    if padsize > (before_pad+after_pad):
        after_pad += 1
    f0_embeddings_stage2 = torch.cat([torch.tensor(np.zeros((before_pad,32))), f0_embeddings_stage2.cpu(), torch.tensor(np.zeros((after_pad,32)))])
    f1_embeddings_stage2 = torch.cat([torch.tensor(np.zeros((before_pad,32))), f1_embeddings_stage2.cpu(), torch.tensor(np.zeros((after_pad,32)))])
    f2_embeddings_stage2 = torch.cat([torch.tensor(np.zeros((before_pad,32))), f2_embeddings_stage2.cpu(), torch.tensor(np.zeros((after_pad,32)))])
    with autocast():
        with torch.no_grad():
            f0_embeddings_stage2 = f0_embeddings_stage2.unsqueeze(0).unsqueeze(0).cuda().half()
            f1_embeddings_stage2 = f1_embeddings_stage2.unsqueeze(0).unsqueeze(0).cuda().half()
            f2_embeddings_stage2 = f2_embeddings_stage2.unsqueeze(0).unsqueeze(0).cuda().half()
            out1 = (seriesnet_b4f0(f0_embeddings_stage2) + seriesnet_b4f1(f1_embeddings_stage2) + seriesnet_b4f2(f2_embeddings_stage2)) / 6.
            out2 = (seriesnet_b4f0(f0_embeddings_stage2.flip(2)) + seriesnet_b4f1(f1_embeddings_stage2.flip(2)) + seriesnet_b4f2(f2_embeddings_stage2.flip(2))) / 6.
            out = torch.sigmoid((out1+out2).cpu()).squeeze(0).numpy()
    submission_csv_data.append([study+"_negative_exam_for_pe", out[0]])
    submission_csv_data.append([study+"_indeterminate", out[1]])
    submission_csv_data.append([study+"_chronic_pe", out[2]])
    submission_csv_data.append([study+"_acute_and_chronic_pe", out[3]])
    submission_csv_data.append([study+"_central_pe", out[4]])
    submission_csv_data.append([study+"_leftsided_pe", out[5]])
    submission_csv_data.append([study+"_rightsided_pe", out[6]])
    submission_csv_data.append([study+"_rv_lv_ratio_gte_1", out[7]])
    submission_csv_data.append([study+"_rv_lv_ratio_lt_1", out[8]])

df = pd.DataFrame(data=submission_csv_data, columns=['id', 'label'])
def check_consistency(sub, test):
    
    '''
    Checks label consistency and returns the errors
    
    Args:
    sub   = submission dataframe (pandas)
    test  = test.csv dataframe (pandas)
    '''
    
    # EXAM LEVEL
    for i in test['StudyInstanceUID'].unique():
        df_tmp = sub.loc[sub.id.str.contains(i, regex = False)].reset_index(drop = True)
        df_tmp['StudyInstanceUID'] = df_tmp['id'].str.split('_').str[0]
        df_tmp['label_type']       = df_tmp['id'].str.split('_').str[1:].apply(lambda x: '_'.join(x))
        del df_tmp['id']
        if i == test['StudyInstanceUID'].unique()[0]:
            df = df_tmp.copy()
        else:
            df = pd.concat([df, df_tmp], axis = 0)
    df_exam = df.pivot(index = 'StudyInstanceUID', columns = 'label_type', values = 'label')
    
    # IMAGE LEVEL
    df_image = sub.loc[sub.id.isin(test.SOPInstanceUID)].reset_index(drop = True)
    df_image = df_image.merge(test, how = 'left', left_on = 'id', right_on = 'SOPInstanceUID')
    df_image.rename(columns = {"label": "pe_present_on_image"}, inplace = True)
    del df_image['id']
    
    # MERGER
    df = df_exam.merge(df_image, how = 'left', on = 'StudyInstanceUID')
    ids    = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    labels = [c for c in df.columns if c not in ids]
    df = df[ids + labels]
    
    # SPLIT NEGATIVE AND POSITIVE EXAMS
    df['positive_images_in_exam'] = df['StudyInstanceUID'].map(df.groupby(['StudyInstanceUID']).pe_present_on_image.max())
    df_pos = df.loc[df.positive_images_in_exam >  0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]
    
    # CHECKING CONSISTENCY OF POSITIVE EXAM LABELS
    rule1a = df_pos.loc[((df_pos.rv_lv_ratio_lt_1  >  0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 >  0.5)) | 
                        ((df_pos.rv_lv_ratio_lt_1  <= 0.5)  & 
                         (df_pos.rv_lv_ratio_gte_1 <= 0.5))].reset_index(drop = True)
    rule1a['broken_rule'] = '1a'
    rule1b = df_pos.loc[(df_pos.central_pe    <= 0.5) & 
                        (df_pos.rightsided_pe <= 0.5) & 
                        (df_pos.leftsided_pe  <= 0.5)].reset_index(drop = True)
    rule1b['broken_rule'] = '1b'
    rule1c = df_pos.loc[(df_pos.acute_and_chronic_pe > 0.5) & 
                        (df_pos.chronic_pe           > 0.5)].reset_index(drop = True)
    rule1c['broken_rule'] = '1c'
    rule1d = df_pos.loc[(df_pos.indeterminate        > 0.5) | 
                        (df_pos.negative_exam_for_pe > 0.5)].reset_index(drop = True)
    rule1d['broken_rule'] = '1d'

    # CHECKING CONSISTENCY OF NEGATIVE EXAM LABELS
    rule2a = df_neg.loc[((df_neg.indeterminate        >  0.5)  & 
                         (df_neg.negative_exam_for_pe >  0.5)) | 
                        ((df_neg.indeterminate        <= 0.5)  & 
                         (df_neg.negative_exam_for_pe <= 0.5))].reset_index(drop = True)
    rule2a['broken_rule'] = '2a'
    rule2b = df_neg.loc[(df_neg.rv_lv_ratio_lt_1     > 0.5) | 
                        (df_neg.rv_lv_ratio_gte_1    > 0.5) |
                        (df_neg.central_pe           > 0.5) | 
                        (df_neg.rightsided_pe        > 0.5) | 
                        (df_neg.leftsided_pe         > 0.5) |
                        (df_neg.acute_and_chronic_pe > 0.5) | 
                        (df_neg.chronic_pe           > 0.5)].reset_index(drop = True)
    rule2b['broken_rule'] = '2b'
    
    # MERGING INCONSISTENT PREDICTIONS
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis = 0)
    
    # OUTPUT
    print('Found', len(errors), 'inconsistent predictions')
    return errors

def neg_ind_pos(exam_pred):
    p = exam_pred[0:2].copy()                       ### negative_exam_for_pe, indeterminate prediction
    if p[0] > 0.5 and p[1] <= 0.5:                  ### negative_exam_for_pe
        return 0
    elif p[0] > 0.5 and p[1] > 0.5:
        if 0.0736196319*p[0] > 0.09202453988*p[1]:
            return 0                                ### negative_exam_for_pe
        else:
            return 1                                ### indeterminate
    elif p[0] <= 0.5 and p[1] > 0.5:                ### indeterminate
        return 1
    else:
        return 2

### Post-processing
eps = 1e-6

exam_preds = []
exam_ids = []
image_preds = []
image_ids = []
for StudyInstanceUID, grp in tqdm(test_df.groupby('StudyInstanceUID')):
    exam_ids.append(StudyInstanceUID)
    
    study_instances = grp['SOPInstanceUID'].values
    study_df = df[df["id"].isin(study_instances)]
    preds = study_df["label"].values
    for instance in study_df["id"].values:
        image_ids.append(instance)
    idx = np.argmax(preds)
    COLUMNS = ['negative_exam_for_pe','indeterminate','rightsided_pe','leftsided_pe','central_pe',
           'rv_lv_ratio_gte_1','rv_lv_ratio_lt_1','chronic_pe','acute_and_chronic_pe']
    study_exam_names = [StudyInstanceUID+"_"+c for c in COLUMNS]
    exam_pred = df[df["id"].isin(study_exam_names)].label.values
    exam_pred = exam_pred[[0,1,6,5,4,7,8,2,3]]
    c = neg_ind_pos(exam_pred)

    if c == 0:
        ### negative_exam_for_pe
        preds = np.where(preds > 0.5, 0.5-eps, preds)
        for i in [1,2,3,4,5,6,7,8]:
            if exam_pred[i] > 0.5:
                exam_pred[i] = 0.5-eps
    elif c == 1:
        ### indeterminate
        preds = np.where(preds > 0.5, 0.5-eps, preds)
        for i in [0,2,3,4,5,6,7,8]:
            if exam_pred[i] > 0.5:
                exam_pred[i] = 0.5-eps
    else:
        ### positive_exam_for_pe

        ### pe_present_on_image
        if preds[idx] <= 0.5:
            preds[idx] = 0.5+eps

        ### rightsided_pe,leftsided_pe,central_pe
        ri_le_ce = np.argmax(exam_pred[2:5])                                
        if exam_pred[ri_le_ce+2] <= 0.5:
            exam_pred[ri_le_ce+2] = 0.5+eps

        ### rv_lv_ratio_gte_1,rv_lv_ratio_lt_1
        if exam_pred[5] > 0.5 and exam_pred[6] > 0.5:
            if exam_pred[5] > exam_pred[6]:
                exam_pred[6] = 0.5-eps
            else:
                exam_pred[5] = 0.5-eps
        elif exam_pred[5] <= 0.5 and exam_pred[6] <= 0.5:
            if exam_pred[5] > exam_pred[6]:
                exam_pred[5] = 0.5+eps
            else:
                exam_pred[6] = 0.5+eps

        ### chronic_pe,acute_and_chronic_pe
        if exam_pred[7] > 0.5 and exam_pred[8] > 0.5:
            if exam_pred[7] > exam_pred[8]:
                exam_pred[8] = 0.5-eps
            else:
                exam_pred[7] = 0.5-eps
    image_preds.append(preds)
    exam_preds.append(exam_pred)
image_ids = np.array(image_ids)
image_preds = np.concatenate(image_preds).astype(np.float64)

exam_ids = np.array(exam_ids)
exam_preds = np.array(exam_preds, dtype=np.float64)

ids = []
labels = []
for StudyInstanceUID, preds in zip(exam_ids, exam_preds):
    for col, pred in zip(COLUMNS, preds):
        ids.append('{}_{}'.format(StudyInstanceUID, col))
        labels.append(pred)
for SOPInstanceUID, pred in zip(image_ids, image_preds):
    ids.append(SOPInstanceUID)
    labels.append(pred)

sub_df = pd.DataFrame()
sub_df['id'] = np.array(ids)
sub_df['label'] = np.array(labels)

check_consistency(sub_df, test_df)
sub_df.to_csv("submission.csv", index=False)