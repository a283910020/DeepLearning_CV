from torch import nn
import torch
from torch.nn.parameter import Parameter


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc  # 输入通道数 --> 3
        self.output_nc = output_nc  # 输出通道数 --> 3
        self.ngf = ngf  # 第一层卷积后的通道数 --> 64
        self.n_blocks = n_blocks  # 残差块数 --> 6
        self.img_size = img_size  # 图像size --> 256
        self.light = light  # 是否使用轻量级模型

        DownBlock = []
        # 先通过一个卷积核尺寸为7的卷积层，图片大小不变，通道数变为64
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling --> 下采样模块
        n_downsampling = 2
        # 两层下采样，img_size缩小4倍（64），通道数扩大4倍（256）
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck  --> 编码器中的残差模块
        mult = 2 ** n_downsampling
        # 6个残差块，尺寸和通道数都不变
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map --> 产生类别激活图
        # 接着global average pooling后的全连接层
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        # 接着global max pooling后的全连接层
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        # 下面1x1卷积和激活函数，是为了得到两个pooling合并后的特征图
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block --> 生成自适应 L-B Normalization(AdaILN)中的Gamma, Beta
        if self.light:  # 确定轻量级，FC使用的是两个256 --> 256的全连接层
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            # 不是轻量级，则下面的1024x1024 --> 256的全连接层和一个256 --> 256的全连接层
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  # (1024x1014, 64x4) crazy
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        # AdaILN中的Gamma, Beta
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck --> 解码器中的自适应残差模块
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling --> 解码器中的上采样模块
        UpBlock2 = []
        # 上采样与编码器的下采样对应
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),  # 注:只有自适应残差块使用AdaILN
                         nn.ReLU(True)]
        # 最后一层卷积层，与最开始的卷积层对应
        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)  # 编码器整个模块
        self.FC = nn.Sequential(*FC)  # 生成gamma,beta的全连接层模块
        self.UpBlock2 = nn.Sequential(*UpBlock2)  # 只包含上采样后的模块，不包含残差块

    def forward(self, input):
        x = self.DownBlock(input)  # 得到编码器的输出,对应途中encoder feature map

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # 全局平均池化
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))  # gap的预测
        gap_weight = list(self.gap_fc.parameters())[0]  # self.gap_fc的权重参数
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)  # 得到全局平均池化加持权重的特征图

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)  # 全局最大池化
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))  # gmp的预测
        gmp_weight = list(self.gmp_fc.parameters())[0]  # self.gmp_fc的权重参数
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)  # 得到全局最大池化加持权重的特征图

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)  # 结合gap和gmp的cam_logit预测
        x = torch.cat([gap, gmp], 1)  # 结合两种池化后的特征图，通道数512
        x = self.relu(self.conv1x1(x))  # 接入一个卷积层，通道数512转换为256

        heatmap = torch.sum(x, dim=1, keepdim=True)  # 得到注意力热力图

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # 轻量级则先经过一个gap
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)  # 得到自适应gamma和beta

        for i in range(self.n_blocks):
            # 将自适应gamma和beta送入到AdaILN
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)  # 通过上采样后的模块，得到生成结果

        return out, cam_logit, heatmap  # 模型输出为生成结果，cam预测以及热力图


class ResnetBlock(nn.Module):  # 编码器中的残差块
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):  # 解码器中的自适应残差块
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out


class adaILN(nn.Module):  # Adaptive Layer-Instance Normalization代码
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        # adaILN的参数p，通过这个参数来动态调整LN和IN的占比
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        # 先求两种规范化的值
        in_mean, in_var = torch.mean(torch.mean(input, dim=2, keepdim=True), dim=3, keepdim=True), torch.var(
            torch.var(input, dim=2, keepdim=True), dim=3, keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(torch.mean(torch.mean(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3,
                                     keepdim=True), torch.var(
            torch.var(torch.var(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        # 合并两种规范化(IN, LN)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                    1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        # 扩张得到结果
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):  # 没有加入自适应的Layer-Instance Normalization，用于上采样
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(torch.mean(input, dim=2, keepdim=True), dim=3, keepdim=True), torch.var(
            torch.var(input, dim=2, keepdim=True), dim=3, keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(torch.mean(torch.mean(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3,
                                     keepdim=True), torch.var(
            torch.var(torch.var(input, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                    1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out