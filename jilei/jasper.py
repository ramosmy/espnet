"""
    Citing from jasper from Nvidia
"""

import torch
import torch.nn as nn
import torch.functional as F
import toml


# ensure the dimension after convolution always the same
def get_same_padding(kernel_size, stride=1, dilation=1):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")

    return (kernel_size // 2) * dilation


def get_model_definition(config_path):
    cfg = {}
    for key, value in toml.load(config_path).items():
        cfg[key] = value

    return cfg


class SubBlock(nn.Module):
    def __init__(self, dropout, in_channels, out_channels, kernel_size, **kwargs):
        super(SubBlock, self).__init__()
        padding = get_same_padding(kernel_size[0])
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding)
        # self.conv = nn.Conv1d(in_channels=256, out_channels=256,
        #                       kernel_size=11, stride=1, padding=5)
        self.batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    """
    input shape: B × F × T
        B: Batch
        F: Features (equal to in_channels in Conv1d)
        T: Time (or Sequence length)
    """
    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        print(x.shape)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, repeat, dropout, in_channels, out_channels, kernel_size):
        super(Block, self).__init__()
        self.repeat = repeat

        # Block内部channel数相同，都等于in_channels
        self.subblock = SubBlock(in_channels=in_channels, out_channels=in_channels,
                                 dropout=dropout, kernel_size=kernel_size)

        # Block内部对最后一层输入进行残差连接前的卷积层用来调整输出的通道数
        self.last_layer_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(num_features=in_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        last_input_list = []
        for _ in range(self.repeat-1):

            # prepare input for the last sub-layer
            last_input = self.last_layer_conv(x)
            last_input = self.batch_norm(last_input)
            last_input_list.append(last_input)

            # connnet inner sub-blocks
            x = self.subblock(x)

        last_input = torch.Tensor(sum(last_input_list))
        last_input = self.activation(last_input)
        last_input = self.dropout(last_input)
        return last_input


class Jasper(nn.Module):
    def __init__(self):
        super(Jasper,self).__init__()
        encoder_layers = []
        # TODO: make feat_in adaptive to init
        feat_in = 256
        cfg = get_model_definition(config_path="./jasper_config.toml")
        for lcfg in cfg["jasper"]:
            block = Block(in_channels=feat_in,
                          out_channels=lcfg['filters'],
                          repeat=lcfg['repeat'],
                          kernel_size=lcfg['kernel'],
                          dropout=lcfg['dropout'],)
            feat_in = lcfg['filters']
            encoder_layers.append(block)
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        s_input= self.encoder(x)
        return s_input


if __name__=='__main__':
    model = Jasper()
    B = 16
    F = 256  # should adapt to the config file
    T = 16000
    input = torch.Tensor(torch.randn(B, F, T))
    for param in model.named_parameters():
        print(param[1].shape, param[0])
    print(input.shape)
    output = model(input)
    print(output.shape)