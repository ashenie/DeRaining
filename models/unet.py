import torch
import torch.nn as nn
import math

# ======================
# ResBlock（带时间）
# ======================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.shortcut(x)


# ======================
# Attention
# ======================
class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).reshape(B, C, -1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1)

        attn = torch.bmm(q.permute(0,2,1), k) * (C ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


# ======================
# UNet（🔥多Attention版）
# ======================
class UNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=3, base=64):
        super().__init__()

        # down
        self.conv0 = nn.Conv2d(in_channels, base, 3, padding=1)

        self.down1 = ResBlock(base, base*2)
        #self.attn1 = AttentionBlock(base*2)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResBlock(base*2, base*4)
        #self.attn2 = AttentionBlock(base*4)
        self.pool2 = nn.MaxPool2d(2)

        # mid
        self.mid1 = ResBlock(base*4, base*4)
        self.attn_mid = AttentionBlock(base*4)
        self.mid2 = ResBlock(base*4, base*4)

        # up
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.up_block1 = ResBlock(base*6, base*2)
        #self.attn_up1 = AttentionBlock(base*2)

        self.up2 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.up_block2 = ResBlock(base*3, base)
        #self.attn_up2 = AttentionBlock(base)

        self.out = nn.Conv2d(base, 4, 1)

    def forward(self, x):
       

        x0 = self.conv0(x)

        x1 = self.down1(x0)
        #x1 = self.attn1(x1)
        x1p = self.pool1(x1)

        x2 = self.down2(x1p)
        #x2 = self.attn2(x2)
        x2p = self.pool2(x2)

        x_mid = self.mid1(x2p)
        x_mid = self.attn_mid(x_mid)
        x_mid = self.mid2(x_mid)

        x = self.up1(x_mid)
        x = torch.cat([x, x2], dim=1)
        x = self.up_block1(x)
        #x = self.attn_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_block2(x)
        #x = self.attn_up2(x)
        out = self.out(x)
        residual = out[:, :3]
        rain_mask = torch.sigmoid(out[:, 3:4])

        return residual, rain_mask
        