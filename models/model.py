import torch.nn as nn
from torch import einsum
from einops import rearrange
# from .resnet import resnet50
from .bert import Bert
# import torchvision.models as models
import torch
import os
import cv2
import math
# from .Mtransformer import Mtransformer
from .pfm import PFM
# from .bi_lstm import BiLSTM
import torch.nn.functional as F
import pickle
import numpy as np
from .modeling import VisionTransformer, CONFIGS

class Attention(nn.Module):
    # attention
    def __init__(self, dim, heads=8, dim_head=96, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim=-1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,labels):
        label_mask = labels.unsqueeze(1)
        mask = (label_mask==labels).int()

        b, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> h b d', h=h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('h i d, h j d -> h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算
        attn1 = attn*(mask.unsqueeze(0))
        out = einsum('h i j, h j d -> h i d', attn1, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'h i d -> i (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）


class Attention_img(nn.Module):
    # attention
    def __init__(self, dim, heads=8, dim_head=96, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim=-1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
     
        b, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> h b d', h=h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('h i d, h j d -> h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算
        out = einsum('h i j, h j d -> h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'h i d -> i (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class IBN(nn.Module):
    def __init__(self,dim):
        super(IBN,self).__init__()
        half1=int(dim/2)
        self.half = half1
        half2 = dim-half1
        self.IN =nn.InstanceNorm1d(half1)
        self.BN =nn.BatchNorm1d(half2)
    def forward(self,x):
        split = torch.split(x,self.half,1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1,out2),1)
        return out

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        config = CONFIGS[args.model_type]
        self.image_model = VisionTransformer(config)
        self.image_model.load_from(np.load(args.pretrain_dir))
        self.language_model = Bert()
        self.PFM = PFM(args)
        self.IBN = IBN(dim=768)
        self.attn = nn.MultiheadAttention(768, 8)

    def forward(self, images, tokens, segments, input_masks,labels):
        img_feat = self.image_model(images)[0]
        txt_feat = self.language_model(tokens, segments, input_masks)[0]
        img_global = self.IBN(img_feat[:, 0, :])
        txt_global = self.IBN(txt_feat[:, 0, :])
        if self.training:
            img_feats, txt_feats, img_o, txt_o,img_rest,txt_rest,imgP,txtP = self.PFM(img_feat, txt_feat)#共有特征提取模
            img_o = self.attn(query =img_o[:,0],
            key = img_o[:,0],
            value = img_o[:,0],
            key_padding_mask = None)[0]
            img_o = self.IBN(img_o)            

        else:
            img_o, txt_o = self.PFM(img_feat, txt_feat)
            img_o = self.IBN(img_o[:,0])

        label_mask = labels.unsqueeze(1)
        mask = (label_mask ==labels).float()
        txt_o = self.attn(query=txt_o[:, 0],
                       key=txt_o[:, 0],
                       value=txt_o[:, 0],
                       key_padding_mask=None,
                       attn_mask=mask)[0]
        txt_o = self.IBN(txt_o)
       
        out_img = torch.cat((img_global.unsqueeze(1), img_o.unsqueeze(1)), dim=1)
        out_txt = torch.cat((txt_global.unsqueeze(1), txt_o.unsqueeze(1)), dim=1)


        if self.training:
              return img_feats, txt_feats,img_rest,txt_rest,imgP,txtP,out_img,out_txt
        else:
              return out_img, out_txt
