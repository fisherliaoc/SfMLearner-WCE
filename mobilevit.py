import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

fan_mode = 'fan_in'


class Refine_M(nn.Module):
    def __init__(self, input_size0, input_size1):
        super(Refine_M, self).__init__()
        c0 = input_size0
        cc0 = input_size1
        self.conv1 = nn.Conv2d(c0 + cc0, max(4, cc0), 3, 1, 1)
        self.conv2 = nn.Conv2d(c0 + cc0, max(4, cc0), 3, 1, 1)
        self.conv3 = nn.Conv2d(max(4, cc0), max(4, cc0), 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * max(4, cc0), c0, 3, 1, 1)
        self.conv5 = nn.Conv2d(2 * c0, c0, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(layer.weight, mode=fan_mode, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for layer in [self.conv4, self.conv5]:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x0, x1):
        _, _, h, w = x1.size()
        upsampled_x0 = F.interpolate(x0, (h, w), mode='bilinear', align_corners=False)
        conv_input = torch.cat((upsampled_x0, x1), 1)
        conv_output = self.relu(self.conv1(conv_input))
        conv_input = self.relu(self.conv2(conv_input))
        conv_input2 = self.relu(self.conv3(conv_input))
        conv_output = torch.cat((conv_output, conv_input2), 1)
        return self.conv5(torch.cat((upsampled_x0, self.conv4(conv_output)), 1))


def conv_1x1_bn(inp, oup, norm_enable=False):
    layers = [nn.Conv2d(inp, oup, 1, 1, 0, bias=not norm_enable)]
    if norm_enable:
        layers.append(nn.BatchNorm2d(oup, affine=True))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1, padding=1, norm_enable=False):
    layers = [nn.Conv2d(inp, oup, kernal_size, stride, padding, bias=not norm_enable)]
    if norm_enable:
        layers.append(nn.BatchNorm2d(oup, affine=True))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_dropout=0., dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(ffn_dropout) if ffn_dropout > 0. else nn.Identity(),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode=fan_mode, nonlinearity='relu')
        if self.net[0].bias is not None:
            nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_normal_(self.net[3].weight, gain=1.0)
        if self.net[3].bias is not None:
            nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, attn_dropout=0., dropout=0.):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0. else nn.Identity()
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        )
        nn.init.xavier_normal_(self.to_qkv.weight, gain=1.0)
        if self.to_qkv.bias is not None:
            nn.init.zeros_(self.to_qkv.bias)
        nn.init.xavier_normal_(self.to_out[0].weight, gain=1.0)
        if self.to_out[0].bias is not None:
            nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_dropout=0., ffn_dropout=0., dropout=0., norm_enable=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm_enable = norm_enable
        if norm_enable:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads, dim_head, attn_dropout, dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, ffn_dropout, dropout))
                ]))
            self.norm = nn.LayerNorm(dim, elementwise_affine=True)
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads, dim_head, attn_dropout, dropout),
                    FeedForward(dim, mlp_dim, ffn_dropout, dropout)
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        if self.norm_enable:
            return self.norm(x)
        else:
            return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, o_channel, kernel_size, patch_size, mlp_dim, attn_dropout=0., ffn_dropout=0., dropout=0., norm_enable=False):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size, norm_enable=norm_enable)
        self.conv2 = conv_1x1_bn(channel, dim, norm_enable=norm_enable)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, attn_dropout, ffn_dropout, dropout, norm_enable=True)
        self.conv3 = conv_1x1_bn(dim, channel, norm_enable=norm_enable)
        self.conv4 = conv_nxn_bn(2 * channel, o_channel, kernel_size, stride=2, norm_enable=norm_enable)

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(conv[0].weight, mode=fan_mode, nonlinearity='relu')
            if norm_enable:
                if conv[1].weight is not None:
                    nn.init.ones_(conv[1].weight)
                    nn.init.zeros_(conv[1].bias)
            else:
                nn.init.zeros_(conv[0].bias)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, input_image_c=3, expansion=4, kernel_size=3, patch_size=(8, 8)):
        super().__init__()
        ih, iw = image_size
        assert ih % patch_size[0] == 0 and iw % patch_size[1] == 0
        norm_enable_global = False

        self.conv1 = conv_nxn_bn(input_image_c * 2, channels[0], kernal_size=7, stride=2, padding=3, norm_enable=norm_enable_global)
        self.mv0 = conv_nxn_bn(channels[0], channels[2], kernal_size=5, stride=2, padding=2, norm_enable=norm_enable_global)
        self.mv1 = conv_nxn_bn(channels[2], channels[4], kernal_size=5, stride=2, padding=2, norm_enable=norm_enable_global)

        for conv in [self.conv1, self.mv0, self.mv1]:
            nn.init.kaiming_normal_(conv[0].weight, mode=fan_mode, nonlinearity='relu')
            if norm_enable_global:
                if conv[1].weight is not None:
                    nn.init.ones_(conv[1].weight)
                    nn.init.zeros_(conv[1].bias)
            else:
                nn.init.zeros_(conv[0].bias)

        self.mvit = nn.ModuleList([
            conv_nxn_bn(channels[5], channels[6], kernal_size=5, stride=2, padding=2, norm_enable=norm_enable_global),
            conv_nxn_bn(channels[7], channels[8], kernal_size=5, stride=2, padding=2, norm_enable=norm_enable_global),
            MobileViTBlock(int(channels[8] * 1.5), 1, channels[8], channels[9], kernel_size, (2, 2), int(channels[8] * 2), norm_enable=norm_enable_global)
        ])

        for conv in self.mvit:
            nn.init.kaiming_normal_(conv[0].weight, mode=fan_mode, nonlinearity='relu')
            if norm_enable_global:
                if conv[1].weight is not None:
                    nn.init.ones_(conv[1].weight)
                    nn.init.zeros_(conv[1].bias)
            else:
                nn.init.zeros_(conv[0].bias)

        mask_c = 4
        self.rm0 = Refine_M(mask_c, channels[9])
        self.rm1 = Refine_M(mask_c, channels[8])
        self.rm2 = Refine_M(mask_c, channels[6])
        self.rm3 = Refine_M(mask_c, channels[4])
        self.rm4 = Refine_M(mask_c, channels[2])
        self.rm5 = Refine_M(mask_c, channels[0])
        self.rm6 = Refine_M(mask_c, input_image_c * 2)

        self.pool = nn.AvgPool2d(ih // 64, 1)
        self.fc = nn.Conv2d(channels[9], num_classes - 5, 1, 1, 0, bias=False)
        self.fc_intri_0 = nn.Conv2d(channels[9], 2, 1, 1, 0)
        self.fc_intri_1 = nn.Conv2d(channels[9], 2, 1, 1, 0)
        self.fc_intri_2 = nn.Conv2d(channels[9], 1, 1, 1, 0)
        self.conv3 = nn.Conv2d(num_classes - 5, mask_c, 1, 1, 0)

        for layer in [self.fc_intri_0, self.fc_intri_2, self.fc, self.fc_intri_1, self.conv3]:
            nn.init.kaiming_normal_(layer.weight, mode=fan_mode, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        o1_x = self.conv1(x)
        o2_x = self.mv0(o1_x)
        o3_x = self.mv1(o2_x)
        o4_x = self.mvit[0](o3_x)
        o5_x = self.mvit[1](o4_x)
        b_pre = self.mvit[2](o5_x)
        b = self.pool(b_pre)
        f_x = self.fc(b)
        f_y0 = self.fc_intri_0(b)
        f_y1 = self.fc_intri_1(b)
        f_y2 = self.fc_intri_2(b)
        fc_out = torch.cat((f_x, f_y0, f_y1, f_y2), 1)
        mask_f = self.conv3(f_x)
        o0_rm = self.rm0(mask_f, b_pre)
        o1_rm = self.rm1(o0_rm, o5_x)
        o2_rm = self.rm2(o1_rm, o4_x)
        o3_rm = self.rm3(o2_rm, o3_x)
        o4_rm = self.rm4(o3_rm, o2_x)
        o5_rm = self.rm5(o4_rm, o1_x)
        o6_rm = self.rm6(o5_rm, x)
        return fc_out.squeeze(-1).squeeze(-1), o4_rm, o5_rm, o6_rm


def mobilevit_xxs(image_size=(256, 256), num_classes=1000, input_image_c=3):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(image_size, dims, channels, num_classes=num_classes, input_image_c=input_image_c, expansion=2)


def mobilevit_xs(image_size=(256, 256), num_classes=1000, input_image_c=3):
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    dims = [144, 192, 240]
    return MobileViT(image_size, dims, channels, num_classes=num_classes, input_image_c=input_image_c)


def mobilevit_s(image_size=(256, 256), num_classes=1000, input_image_c=3):
    dims = [144, 192, 240]
    channels = [32, 32, 64, 64, 128, 128, 256, 256, 256, 512, 640]
    return MobileViT(image_size, dims, channels, num_classes=num_classes, input_image_c=input_image_c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0


if __name__ == '__main__':
    img1 = torch.randn(5, 4, 256, 256)
    img2 = torch.randn(5, 4, 256, 256)

    for model_func in [mobilevit_xxs, mobilevit_xs, mobilevit_s]:
        model = model_func()
        fc_out, o4_rm, o6_rm = model(img1, img2)
        print(f"Model: {model_func.__name__}")
        print(f"fc_out shape: {fc_out.shape}")
        print(f"o4_rm shape: {o4_rm.shape}")
        print(f"o6_rm shape: {o6_rm.shape}")
        print(f"Total params: {count_parameters(model):.2f}M\n")
