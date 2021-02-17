import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

from lib.model.minkowski.ME_layers import get_norm_layer, get_res_block
from lib.utils import kabsch_transformation_estimation

_EPS = 1e-6

class SparseEnoder(ME.MinkowskiNetwork):
    CHANNELS = [None, 64, 64, 128, 128]

    def __init__(self,
                in_channels=3,
                out_channels=128,
                bn_momentum=0.1,
                conv1_kernel_size=9,
                norm_type='IN',
                D=3):

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        CHANNELS = self.CHANNELS

    
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm_layer(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_res_block(
                BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm_layer(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_res_block(
                BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm_layer(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_res_block(
                BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)



    def forward(self, x, tgt_feature=False):

        skip_features = []
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out = self.block1(out_s1)

        skip_features.append(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out = self.block2(out_s2)

        skip_features.append(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out = self.block3(out_s4)

        skip_features.append(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out = self.block4(out_s8)

        return out, skip_features





class SparseDecoder(ME.MinkowskiNetwork):
    TR_CHANNELS = [None, 64, 128, 128, 128]
    CHANNELS = [None, 64, 64, 128, 128]

    def __init__(self,
                out_channels=128,
                bn_momentum=0.1,
                norm_type='IN',
                D=3):

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        TR_CHANNELS = self.TR_CHANNELS
        CHANNELS = self.CHANNELS


        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm4_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_res_block(
                BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)


        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
                in_channels=CHANNELS[3] + TR_CHANNELS[4],
                out_channels=TR_CHANNELS[3],
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D)
        self.norm3_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_res_block(
                BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)


        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
                in_channels=CHANNELS[2] + TR_CHANNELS[3],
                out_channels=TR_CHANNELS[2],
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D)
        self.norm2_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_res_block(
                BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)



        self.conv1_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)


    def forward(self, x, skip_features):
        
        out = self.conv4_tr(x)
        out = self.norm4_tr(out)
        
        out_s4_tr = self.block4_tr(out)

        out = ME.cat(out_s4_tr, skip_features[-1])

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out_s2_tr = self.block3_tr(out)
        
        out = ME.cat(out_s2_tr, skip_features[-2])

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out_s1_tr = self.block2_tr(out)
        
        out = ME.cat(out_s1_tr, skip_features[-3])

        out = self.conv1_tr(out)
        out = MEF.relu(out)
        out = self.final(out)

        return out

class SparseFlowRefiner(ME.MinkowskiNetwork):
    BLOCK_NORM_TYPE = 'BN'
    NORM_TYPE = 'BN'

    def __init__(self,
                flow_dim = 3,
                flow_channels = 64,
                out_channels=3,
                bn_momentum=0.1,
                conv1_kernel_size=5,
                D=3):

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=flow_dim,
            out_channels=flow_channels,
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=flow_channels,
            out_channels=flow_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)



        self.conv3 = ME.MinkowskiConvolution(
            in_channels=flow_channels,
            out_channels=flow_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=flow_channels,
            out_channels=flow_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=flow_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)


    def forward(self, flow):

        
        out =  MEF.relu(self.conv1(flow))
        out =  MEF.relu(self.conv2(out))

        out = MEF.relu(self.conv3(out))
        out = MEF.relu(self.conv4(out))

        res_flow = self.final(out)
        

        return flow + res_flow


class EgoMotionHead(nn.Module):
    """
    Class defining EgoMotionHead
    """

    def __init__(self, add_slack=True, sinkhorn_iter=5):
        nn.Module.__init__(self)

        self.slack = add_slack
        self.sinkhorn_iter = sinkhorn_iter

        # Affinity parameters
        self.beta = torch.nn.Parameter(torch.tensor(-5.0))
        self.alpha = torch.nn.Parameter(torch.tensor(-5.0))

        self.softplus = torch.nn.Softplus()


    def compute_rigid_transform(self, xyz_s, xyz_t, weights):
        """Compute rigid transforms between two point sets

        Args:
            a (torch.Tensor): (B, M, 3) points
            b (torch.Tensor): (B, N, 3) points
            weights (torch.Tensor): (B, M)

        Returns:
            Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
        """

        weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
        centroid_s = torch.sum(xyz_s * weights_normalized, dim=1)
        centroid_t = torch.sum(xyz_t * weights_normalized, dim=1)
        s_centered = xyz_s - centroid_s[:, None, :]
        t_centered = xyz_t - centroid_t[:, None, :]
        cov = s_centered.transpose(-2, -1) @ (t_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_s[:, :, None] + centroid_t[:, :, None]

        transform = torch.cat((rot_mat, translation), dim=2)

        return transform

    def sinkhorn(self, log_alpha, n_iters=5, slack=True):
        """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)
        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        """

        # Sinkhorn iterations

        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)


        log_alpha = log_alpha_padded[:, :-1, :-1]

        return log_alpha


    def forward(self, score_matrix, mask, xyz_s, xyz_t):

        affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)

         # Compute weighted coordinates
        log_perm_matrix = self.sinkhorn(affinity, n_iters=self.sinkhorn_iter, slack=self.slack)

        perm_matrix = torch.exp(log_perm_matrix) * mask
        weighted_t = perm_matrix @ xyz_t / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

        # Compute transform and transform points
        #transform = self.compute_rigid_transform(xyz_s, weighted_t, weights=torch.sum(perm_matrix, dim=2))
        R_est, t_est, _, _ = kabsch_transformation_estimation(xyz_s, weighted_t, weights=torch.sum(perm_matrix, dim=2))
        return R_est, t_est, perm_matrix



class SparseSegHead(ME.MinkowskiNetwork):

    def __init__(self,
                in_channels=64,
                out_channels=128,
                bn_momentum=0.1,
                norm_type='IN',
                D=3):

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type

        self.seg_head_1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

        self.norm_1 = get_norm_layer(NORM_TYPE, in_channels, bn_momentum=bn_momentum, D=D)

        self.seg_head_2 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)


    def forward(self, x):
        
        out = self.seg_head_1(x)
        out = self.norm_1(out)
        out = MEF.relu(out)

        out = self.seg_head_2(out)


        return out