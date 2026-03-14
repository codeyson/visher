"""
model_loader.py
Exact RawNet architecture from model.py + model_config_RawNet.yaml.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────
# SincConv  (fixed filterbank, no learned weights)
# ─────────────────────────────────────────────────────────────

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size,
                 in_channels=1, sample_rate=24000,
                 stride=1, padding=0, dilation=1,
                 bias=False, groups=1):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports in_channels=1")

        self.out_channels = out_channels
        self.kernel_size  = kernel_size + (1 if kernel_size % 2 == 0 else 0)
        self.sample_rate  = sample_rate
        self.device       = device
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation

        NFFT = 512
        f    = int(sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(fmel.min(), fmel.max(), out_channels + 1)
        self.mel   = self.to_hz(filbandwidthsmel)
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                   (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin  = self.mel[i]
            fmax  = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                        2 * fmax * self.hsupp.numpy() / self.sample_rate)
            hLow  = (2 * fmin / self.sample_rate) * np.sinc(
                        2 * fmin * self.hsupp.numpy() / self.sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = (Tensor(np.hamming(self.kernel_size))
                                    * Tensor(hideal))

        filters = self.band_pass.to(x.device).view(
            self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


# ─────────────────────────────────────────────────────────────
# Residual block
# ─────────────────────────────────────────────────────────────

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1, stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample     = True
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1],
                                             kernel_size=1, padding=0, stride=1)
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        out  = self.mp(out)
        return out


# ─────────────────────────────────────────────────────────────
# RawNet  (exact copy of model.py)
# ─────────────────────────────────────────────────────────────

class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super().__init__()
        self.device = device

        self.Sinc_conv = SincConv(
            device      = device,
            out_channels= d_args['filts'][0],
            kernel_size = d_args['first_conv'],
            in_channels = d_args['in_channels'],
        )

        self.first_bn = nn.BatchNorm1d(d_args['filts'][0])
        self.selu     = nn.SELU(inplace=True)

        self.block0 = nn.Sequential(Residual_block(d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(d_args['filts'][2]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        def _attn(n): return nn.Sequential(nn.Linear(n, n))
        self.fc_attention0 = _attn(d_args['filts'][1][-1])
        self.fc_attention1 = _attn(d_args['filts'][1][-1])
        self.fc_attention2 = _attn(d_args['filts'][2][-1])
        self.fc_attention3 = _attn(d_args['filts'][2][-1])
        self.fc_attention4 = _attn(d_args['filts'][2][-1])
        self.fc_attention5 = _attn(d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(d_args['filts'][2][-1])
        self.gru = nn.GRU(
            input_size = d_args['filts'][2][-1],
            hidden_size= d_args['gru_node'],
            num_layers = d_args['nb_gru_layer'],
            batch_first= True,
        )

        self.fc1_binary_gru = nn.Linear(d_args['gru_node'],    d_args['nb_fc_node'])
        self.fc2_binary_gru = nn.Linear(d_args['nb_fc_node'],  2, bias=True)
        self.fc1_multi_gru  = nn.Linear(d_args['gru_node'],    d_args['nb_fc_node'])
        self.fc2_multi_gru  = nn.Linear(d_args['nb_fc_node'],  7, bias=True)

        self.sig          = nn.Sigmoid()
        self.logsoftmax   = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: (batch, time)  — same as training
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)          # → (batch, 1, time)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        # block + attention (formula: x*y + y  from original model.py)
        x0 = self.block0(x)
        y0 = self.sig(self.fc_attention0(self.avgpool(x0).view(x0.size(0), -1)))
        y0 = y0.view(y0.size(0), y0.size(1), -1)
        x  = x0 * y0 + y0

        x1 = self.block1(x)
        y1 = self.sig(self.fc_attention1(self.avgpool(x1).view(x1.size(0), -1)))
        y1 = y1.view(y1.size(0), y1.size(1), -1)
        x  = x1 * y1 + y1

        x2 = self.block2(x)
        y2 = self.sig(self.fc_attention2(self.avgpool(x2).view(x2.size(0), -1)))
        y2 = y2.view(y2.size(0), y2.size(1), -1)
        x  = x2 * y2 + y2

        x3 = self.block3(x)
        y3 = self.sig(self.fc_attention3(self.avgpool(x3).view(x3.size(0), -1)))
        y3 = y3.view(y3.size(0), y3.size(1), -1)
        x  = x3 * y3 + y3

        x4 = self.block4(x)
        y4 = self.sig(self.fc_attention4(self.avgpool(x4).view(x4.size(0), -1)))
        y4 = y4.view(y4.size(0), y4.size(1), -1)
        x  = x4 * y4 + y4

        x5 = self.block5(x)
        y5 = self.sig(self.fc_attention5(self.avgpool(x5).view(x5.size(0), -1)))
        y5 = y5.view(y5.size(0), y5.size(1), -1)
        x  = x5 * y5 + y5

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]

        x_binary       = self.fc2_binary_gru(self.fc1_binary_gru(x))
        output_binary  = self.logsoftmax(x_binary)

        x_multi        = self.fc2_multi_gru(self.fc1_multi_gru(x))
        output_multi   = self.logsoftmax(x_multi)

        return output_binary, output_multi


# ─────────────────────────────────────────────────────────────
# Config  (from model_config_RawNet.yaml)
# ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'nb_samp'     : 64600,
    'first_conv'  : 1024,
    'in_channels' : 1,
    'filts'       : [20, [20, 20], [20, 128], [128, 128]],
    'nb_fc_node'  : 1024,
    'gru_node'    : 1024,
    'nb_gru_layer': 3,
}


def _unwrap(raw):
    if isinstance(raw, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in raw:
                return raw[k]
    return raw


def load_model(checkpoint_path: str,
               config: dict | None = None,
               device: str | None  = None) -> RawNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import copy
    cfg    = copy.deepcopy(config or DEFAULT_CONFIG)
    state  = _unwrap(torch.load(checkpoint_path, map_location=device))

    model  = RawNet(cfg, device).to(device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[model_loader] Missing   : {missing[:5]}")
    if unexpected:
        print(f"[model_loader] Unexpected: {unexpected[:5]}")

    model.eval()
    print(f"[model_loader] Loaded '{checkpoint_path}' on {device}  ✓")
    return model