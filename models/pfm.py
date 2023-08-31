
import torch
from torch import nn, einsum

from einops import rearrange, repeat

import math


def l2norm(X,dim,eps=1e-8):
    norm = torch.pow(X,2).sum(dim=dim,keepdim=True).sqrt()+eps
    X=torch.div(X,norm)
    return X

class IBN(nn.Module):
    def __init__(self,dim):
        super(IBN,self).__init__()
        half1=int(dim/2)
        self.half = half1
        half2 = dim-half1
        self.IN =nn.InstanceNorm2d((1,half1))
        self.BN =nn.BatchNorm2d(half2)
    def forward(self,x):
        split = torch.split(x,self.half,2)
        out1 = self.IN(split[0].unsqueeze(2)).squeeze(2)
        out2 = self.BN(split[1].unsqueeze(2).transpose(1,3)).squeeze(2).transpose(1,2)
        out = torch.cat((out1,out2),2)
        return out
#-----------------------------------------------##########
class CrossAttention(nn.Module):
    def __init__(
            self,
            dim=768,
            num_heads=8,
            qkv_bias=False,
            qk_scale=1 / math.sqrt(768),
            attn_drop=0.0,
            proj_drop=0.0,
            dropout = 0.,
    ):
        super().__init__()
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.to_out = nn.Sequential(
            nn.Linear(768, dim),
            nn.Dropout(dropout)
        )

        self.IBN = IBN(dim=768)


    def forward(self, img, txt, input_masks=None):
        device = torch.device('cuda:0')
        B1, N1, C1 = img.shape
        # img = self.ln1(img)
        qkv1 = self.qkv(img)
        qkv1 = qkv1.reshape(B1, N1, 3, C1).permute(2, 0, 1, 3)
        q1, k1, v1 = (
            qkv1[0],
            qkv1[1],
            qkv1[2],
        )  # make torchscript happy (cannot use tensor as tuple)


        B2, N2, C2 = txt.shape
        # txt = self.ln2(txt)
        qkv2 = self.qkv(txt)
        qkv2 = qkv2.reshape(B2, N2, 3, C2).permute(2, 0, 1, 3)
        q2, k2, v2 = (
            qkv2[0],
            qkv2[1],
            qkv2[2],)

        attn1 = torch.matmul(q2, k1.transpose(1, 2)) * self.scale
        if input_masks is not None:
            extended_attention_mask = input_masks.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=torch.float32
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attn1 = attn1 + extended_attention_mask
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        a, b, c = attn1.shape
        one1 = torch.ones((a, b, c))
        one1 = one1.to(device)
        R_attn1 = torch.sub(one1, attn1)
        imgP = torch.matmul(attn1, v1)
        imgN = torch.matmul(R_attn1, v1)
        imgP = self.IBN(imgP)
        imgN = self.IBN(imgN)

        attn2 = torch.matmul(q1, k2.transpose(1, 2)) * self.scale
        if input_masks is not None:
            extended_attention_mask = input_masks.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=torch.float32
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attn2 = attn2 + extended_attention_mask
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        a, b, c = attn2.shape
        one2 = torch.ones((a, b, c))
        one2 = one2.to(device)
        R_attn2 = torch.sub(one2, attn2)

        txtP = torch.matmul(attn2, v2)
        txtN = torch.matmul(R_attn2, v2)
        txtP = self.IBN(txtP)
        txtN = self.IBN(txtN)

        return imgP,imgN,txtP,txtN

#########----------------------------------------------------############################
class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# FeedForward层：由全连接层，配合激活函数GELU和Dropout实现，对应框图中蓝色的MLP。
# 参数dim和hidden_dim分别是输入输出的维度和中间层的维度，dropour则是dropout操作的概率参数p
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
        self.net = torch.nn.DataParallel(self.net)

    def forward(self, x):
        return self.net(x)


# 注意力层
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

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）

class Transformer(nn.Module):
    def __init__(self, dim=768, depth=1, heads=8, dim_head=96, mlp_dim=768, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                Norm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 多头自注意力模块
                Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前向传播模块
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x


#########################-------------------------------------------#####################
class PFM(nn.Module):
    def __init__(self, args, qk_scale=1/math.sqrt(768)):
        super(PFM, self).__init__()
        self.scale = qk_scale
        self.RA1 = CrossAttention()
        self.RA2 = CrossAttention()
        self.img_transformer1 = Transformer()
        self.txt_transformer1 = Transformer()
        self.IBN=IBN(dim=768)
  
    def forward(self, img_feat, txt_feat):  
        P0 = []
        P1 = []
        if self.training:
            imgP1,imgN1,txtP1,txtN1= self.RA1(img_feat, txt_feat)
            img_feats1 = self.img_transformer1(imgN1)+imgP1
            txt_feats1 = self.txt_transformer1(txtN1)+txtP1
            img_o1 = self.img_transformer1(img_feat)+img_feat
            txt_o1 = self.txt_transformer1(txt_feat)+txt_feat
            img_rP0,img_rest1,img_rP1,_ = self.RA2(img_o1, img_feats1)
            txt_rP0,txt_rest1,txt_rP1,_ = self.RA2(txt_o1, txt_feats1)

            P0.append(imgP1)
            P0.append(img_rP0)
            P0.append(txt_rP0)
            P1.append(txtP1)
            P1.append(img_rP1)
            P1.append(txt_rP1)

            return img_feats1, txt_feats1, img_o1, txt_o1,img_rest1,txt_rest1,P0,P1

        else:
            img_o1 = self.img_transformer1(img_feat)+img_feat
            txt_o1 = self.txt_transformer1(txt_feat)+txt_feat          
           
            return img_o1, txt_o1




