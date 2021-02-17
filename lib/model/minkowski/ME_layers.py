import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

#### NORMALIZATION LAYER ####
def get_norm_layer(norm_type, num_feats, bn_momentum=0.05, D=-1):

    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats)
  
    else:
        raise ValueError(f'Type {norm_type}, not defined')

#### RESIDUAL BLOCK ####

class ResBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self,
                inplanes,
                planes,
                stride=1,
                dilation=1,
                downsample=None,
                bn_momentum=0.1,
                D=3):
        super(ResBlockBase, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)

        self.norm1 = get_norm_layer(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class ResBlockBN(ResBlockBase):
    NORM_TYPE = 'BN'


class ResBlockIN(ResBlockBase):
    NORM_TYPE = 'IN'


def get_res_block(norm_type,
                  inplanes,
                  planes,
                  stride=1,
                  dilation=1,
                  downsample=None,
                  bn_momentum=0.1,
                  D=3):

    if norm_type == 'BN':
        return ResBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  
    elif norm_type == 'IN':
        return ResBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  
    else:
        raise ValueError(f'Type {norm_type}, not defined')
