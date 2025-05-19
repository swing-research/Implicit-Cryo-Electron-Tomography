import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# From https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py#L67


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # if self.kwargs['include_input']:
        #     embed_fns.append(lambda x: x)
        #     out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# Model
class NeRF(nn.Module):
    def __init__(self, embed, D=8, W=256, input_ch=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
    """
        super(NeRF, self).__init__()
        self.embed = embed
        self.D = D
        self.W = W
        self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts = self.embed.embed(x)
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # h = torch.cat([feature, input_views], -1)
            h = feature

            # for i, l in enumerate(self.views_linears):
            #     h = self.views_linears[i](h)
            #     h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple

import torch
import torch.nn as nn


class NeRF2(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        embed,
        pos_dim: int,
        feat_dim: int = 256,
        rgb_dim: int = 1
    ):
        """
        Constructor of class 'NeRF'.

        Args:
            pos_dim (int): Dimensionality of coordinate vectors of sample points.
            view_dir_dim (int): Dimensionality of view direction vectors.
            feat_dim (int): Dimensionality of feature vector within forward propagation.
                Set to 256 by default following the paper.
        """
        super().__init__()

        density_dim = 1
        self.embed = embed
        self._pos_dim = pos_dim
        # self._view_dir_dim = view_dir_dim
        self._feat_dim = feat_dim

        # fully-connected layers
        self.fc_in = nn.Linear(self._pos_dim, self._feat_dim)
        self.fc_1 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_2 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_3 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_4 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_5 = nn.Linear(self._feat_dim + self._pos_dim, self._feat_dim)
        self.fc_6 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_7 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_8 = nn.Linear(self._feat_dim, self._feat_dim)
        self.fc_9 = nn.Linear(self._feat_dim, self._feat_dim // 2)
        self.fc_out = nn.Linear(self._feat_dim // 2, rgb_dim)

        # activation layer
        self.relu_actvn = nn.ReLU()
        self.tanh_actvn = nn.Tanh()

    def forward(
        self,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos (torch.Tensor): Tensor of shape (N, self.pos_dim).
                Coordinates of sample points along rays.
            view_dir (torch.Tensor): Tensor of shape (N, self.dir_dim).
                View direction vectors.

        Returns:
            sigma (torch.Tensor): Tensor of shape (N, ).
                Density at the given sample points.
            rgb (torch.Tensor): Tensor of shape (N, 3).
                Radiance at the given sample points.
        """
        # # check input tensors
        # if (pos.ndim != 2) or (view_dir.ndim != 2):
        #     raise ValueError(f"Expected 2D tensors. Got {pos.ndim}, {view_dir.ndim}-D tensors.")
        # if pos.shape[0] != view_dir.shape[0]:
        #     raise ValueError(
        #         f"The number of samples must match. Got {pos.shape[0]} and {view_dir.shape[0]}."
        #     )
        # if pos.shape[-1] != self._pos_dim:
        #     raise ValueError(f"Expected {self._pos_dim}-D position vector. Got {pos.shape[-1]}.")
        # if view_dir.shape[-1] != self._view_dir_dim:
        #     raise ValueError(
        #         f"Expected {self._view_dir_dim}-D view direction vector. Got {view_dir.shape[-1]}."
        #     )

        pos = self.embed.embed(pos)

        x = self.relu_actvn(self.fc_in(pos))
        x = self.relu_actvn(self.fc_1(x))
        x = self.relu_actvn(self.fc_2(x))
        x = self.relu_actvn(self.fc_3(x))
        x = self.relu_actvn(self.fc_4(x))

        x = torch.cat([pos, x], dim=-1)

        x = self.relu_actvn(self.fc_5(x))
        x = self.relu_actvn(self.fc_6(x))
        x = self.relu_actvn(self.fc_7(x))
        x = self.relu_actvn(self.fc_8(x))
        x = self.relu_actvn(self.fc_9(x))
        rgb = self.tanh_actvn(self.fc_out(x))

        return rgb

    # @property
    # def pos_dim(self) -> int:
    #     """Returns the acceptable dimensionality of coordinate vectors."""
    #     return self._pos_dim
    #
    # @property
    # def view_dir_dim(self) -> int:
    #     """Returns the acceptable dimensionality of view direction vectors."""
    #     return self._view_dir_dim
    #
    # @property
    # def feat_dim(self) -> int:
    #     """Returns the dimensionality of internal feature vectors."""
    #     return self._feat_dim