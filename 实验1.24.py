import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy import ndimage
from scipy.ndimage import zoom
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import logging
import sys
import time
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import SimpleITK as sitk
from medpy import metric
import matplotlib.pyplot as plt  # 用于保存照片

# Model Definition (保持不变)
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SE(inp, hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class NoBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = nn.Conv2d(cin, cmid, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = nn.Conv2d(cmid, cmid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = nn.Conv2d(cmid, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.gelu = nn.GELU()
        if (stride != 1 or cin != cout):
            self.downsample = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)
        self.normal = nn.GroupNorm(1, cin, eps=1e-6)

    def forward(self, x):
        residual = x
        if (self.stride != 1 or self.cin != self.cout):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        x = self.normal(x)
        y = self.gelu(self.gn1(self.conv1(x)))
        out_to_trans = self.gelu(self.gn2(self.conv2(y)))
        y = self.conv3(y)
        y = self.gn3(y)
        y = self.gelu(y + residual)
        return y, out_to_trans

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        heads = inp // dim_head
        project_out = not (heads == 1 and dim_head == inp)
        self.ih, self.iw = image_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))
        coords = torch.meshgrid([torch.arange(self.ih), torch.arange(self.iw)])
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Conv2d(inp, oup, 1),
            nn.Dropout2d(dropout, inplace=True)
        ) if project_out else nn.Identity()
        self.projQ = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )
        self.projK = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )
        self.projV = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )

    def forward(self, x, y=None):
        q = self.projQ(x)
        k = self.projK(x)
        v = self.projV(x)
        q = rearrange(q, 'b c ih iw -> b (ih iw) c')
        k = rearrange(k, 'b c ih iw -> b (ih iw) c')
        v = rearrange(v, 'b c ih iw -> b (ih iw) c')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2))
        if y is not None:
            q_y = self.projQ(y)
            k_y = self.projK(y)
            q_y = rearrange(q_y, 'b c ih iw -> b (ih iw) c')
            k_y = rearrange(k_y, 'b c ih iw -> b (ih iw) c')
            q_y = rearrange(q_y, 'b n (h d) -> b h n d', h=self.heads)
            k_y = rearrange(k_y, 'b n (h d) -> b h n d', h=self.heads)
            dots = (dots + torch.matmul(q_y, k_y.transpose(-1, -2))) * self.scale
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        out = self.to_out(out)
        return out

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=1, embed_dim=32, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        self.ih, self.iw = image_size
        self.layer1 = NoBottleneck(inp, inp, inp)
        self.attn = Attention(inp, inp, image_size, heads, dim_head, dropout)
        self.mlp = CMLP(inp, 4 * inp, drop=dropout)
        self.norm = nn.GroupNorm(1, inp, eps=1e-6)
        self.SA1 = SpatialAttention(7)
        self.conv1x1 = nn.Conv2d(2*inp, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, CONV, TRANS):
        CONV, x_totran = self.layer1(CONV)
        TRANS = self.norm(TRANS)
        TRANS = self.attn(TRANS, x_totran) + TRANS
        TRANS = self.norm(TRANS)
        TRANS = self.mlp(TRANS) + TRANS
        CONV_SA = self.SA1(TRANS) * CONV
        F = torch.cat([CONV_SA, TRANS], dim=1)
        CONV = self.conv1x1(F)
        return CONV, TRANS

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class PCTNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}
        self.s0 = self._make_layer(conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = makelayer(channels[3], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.upsamplex2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_more = Conv2dReLU(channels[3], channels[2], kernel_size=3, padding=1, use_batchnorm=True)
        self.ups2 = DecoderBlock(channels[2], channels[1], channels[2])
        self.ups3 = DecoderBlock(channels[1], channels[0], channels[1])
        self.ups4 = DecoderBlock(channels[0], channels[0], channels[0])
        self.heading = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            Conv2dReLU(channels[0], channels[0] // 2, kernel_size=3, padding=1, use_batchnorm=True),
            nn.Conv2d(channels[0] // 2, 9, kernel_size=1, padding=0))
        self.T_down = PatchEmbed(patch_size=16, in_c=in_channels, embed_dim=channels[3], norm_layer=nn.LayerNorm)
        self.layer3 = nn.Conv2d(channels[2], channels[3], kernel_size=(3, 3), stride=(2, 2), padding=1, groups=1)

    def forward(self, input):
        input = input.repeat(1, 3, 1, 1)
        x = self.s0(input)
        skip1 = x
        x = self.s1(x)
        skip2 = x
        x = self.s2(x)
        skip3 = x
        x_t, H, W = self.T_down(input)
        x_t = rearrange(x_t, 'b (h w) c -> b c h w ', h=H, w=W)
        x = self.layer3(skip3)
        x, x_t = self.s3(x, x_t)
        x = self.conv_more(x+x_t)
        x = self.ups2(x, skip3)
        x = self.ups3(x, skip2)
        x = self.ups4(x, skip1)
        x = self.heading(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

class makelayer(nn.Module):
    def __init__(self, inp, oup, depth, image_size):
        super().__init__()
        self.strat = Transformer(inp, oup, image_size)
        self.blocks = nn.ModuleList([Transformer(oup, oup, image_size) for i in range(depth - 1)])

    def forward(self, CONV, TRANS):
        CONV, TRANS = self.strat(CONV, TRANS)
        for blk in self.blocks:
            CONV, TRANS = blk(CONV, TRANS)
        return CONV, TRANS

def pctnet():
    num_blocks = [2, 2, 6, 6]
    channels = [64, 128, 256, 512]
    return PCTNet((224, 224), 3, num_blocks, channels)

# Dataset and DataLoader
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 定义 RandomGenerator 和 Synapse_dataset
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class gaiDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(gaiDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        TP = torch.sum(score * target)
        FP = torch.sum((1-score) * target)
        FN = torch.sum((1-target) * score)
        a = 1.0
        b = FP / (FP + FN)
        loss = (a * TP + smooth)/ (a * TP + b * FP + (1 - b) * FN + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

# Utility Functions
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred

                # 保存照片
                if test_save_path is not None:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 3, 1)
                    plt.title("Input Image")
                    plt.imshow(slice, cmap='gray')
                    plt.subplot(1, 3, 2)
                    plt.title("Prediction")
                    plt.imshow(pred, cmap='jet', vmin=0, vmax=classes-1)
                    plt.subplot(1, 3, 3)
                    plt.title("Ground Truth")
                    plt.imshow(label[ind], cmap='jet', vmin=0, vmax=classes-1)
                    plt.savefig(os.path.join(test_save_path, f"{case}_slice_{ind}.png"))
                    plt.close()
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

# Training and Inference
def inference(best_performance, best_mean_hd95, epoch_num, args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=8, pin_memory=False)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('epoch : %d idx %d case %s mean_dice %f mean_hd95 %f' % (epoch_num, i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance : mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    if performance > best_performance:
        best_performance = performance
        best_mean_hd95 = mean_hd95
        change = True
    else:
        change = False
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (best_performance, best_mean_hd95))
    return performance, mean_hd95, best_performance, best_mean_hd95,change

def trainer_synapse(args, model, snapshot_path, times, strat=0, gai=False):

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    if gai:
        dice_loss = gaiDiceLoss(num_classes)
    else:
        dice_loss = DiceLoss(num_classes)

    if args.is_reload_path:
        model.load_state_dict(torch.load(snapshot_path+args.reload_path, map_location=torch.device('cpu')))

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    snapshot_name = snapshot_path.split('/')[-1]

    iter_num = strat * len(trainloader)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    best_mean_hd95 = 0.0

    dice_loss_list=[]
    hd_loss_list=[]

    iterator = tqdm(range(strat, max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        a=b=c=0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch, label_batch  # 数据在 CPU 上
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            logging.info('iteration %d : loss : %f, loss_dice : %f, loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))
            a=a+loss.item()
            b=b+loss_dice.item()
            c=c+loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        a=a/len(trainloader)
        b=b/len(trainloader)
        c=c/len(trainloader)
        logging.info('epoch: %d / %d , loss : %f, loss_dice : %f, loss_ce: %f' % (epoch_num, max_epoch, a, b, c))
        save_mode_path = os.path.join(snapshot_path, 'interim.pth')
        torch.save(model.state_dict(), save_mode_path)
        if epoch_num > 500 and (epoch_num + 1) % 25 == 0:
            snapshot = os.path.join(snapshot_path, 'interim.pth')
            model.load_state_dict(torch.load(snapshot))
            if args.is_savenii:
                args.test_save_dir = '../predictions'
                test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
                os.makedirs(test_save_path, exist_ok=True)
            else:
                test_save_path = None
            dice_loss_iter, hd_loss_iter, best_performance, best_mean_hd95, change = inference(best_performance, best_mean_hd95, epoch_num,
                                                                         args, model, test_save_path)
            if change:
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(model.state_dict(), save_mode_path)
            dice_loss_list.append(dice_loss_iter)
            hd_loss_list.append(hd_loss_iter)
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            times_save_mode_path = os.path.join(snapshot_path, str(times) + 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), times_save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    for i in range(len(dice_loss_list)):
        print(i, dice_loss_list[i], hd_loss_list[i])
    writer.close()
    print("Training Finished!")
    return "Training Finished!"

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=r"C:\Users\32572\Desktop\EI\Synapsedata\all\images", help='root dir for data')
    parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str, default=, help='list dir')
    parser.add_argument('--volume_path', type=str, default='./masks', help='root dir for validation volume data')
    parser.add_argument('--is_savenii', default=True, help='whether to save results during inference')
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
    parser.add_argument('--vit_name', type=str, default='PCTNet', help='select one vit model')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument("--reload_path", type=str, default='./interim.pth')
    parser.add_argument('--is_reload_path', default=False)

    args = parser.parse_args()

    if not args.deterministic:

    random.seed(args.seed)   cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    args.exp = 'pctnet_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'pctnet')
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    print("snapshot_path: ", snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    for i in range(1):
        net = pctnet()
        trainer = {'Synapse': trainer_synapse}
        trainer[dataset_name](args, net, snapshot_path, i, 0)

if __name__ == "__main__":
    main()