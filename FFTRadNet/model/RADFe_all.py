import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# from thop import profile, clever_format
# from fvcore.nn import FlopCountAnalysis, flop_count_table

from ptflops import get_model_complexity_info
from einops import repeat, rearrange

from mamba_ssm import Mamba

class ChirpNet1(nn.Module):
    def __init__(self, num_features=192, hidden_dim=1024, num_layers=1, linear_dims=[4, 8], conv_channels = 32):
        super(ChirpNet1, self).__init__()
        channels = 32
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Linear layers before GRU
        self.linears = nn.ModuleList()
        last_dim = num_features
        for next_dim in linear_dims:
            self.linears.append(nn.Linear(last_dim * channels, next_dim * channels))
            last_dim = next_dim
        
        last_dim = last_dim * channels

        # GRU layer
        self.gru = nn.GRU(last_dim, self.hidden_dim, num_layers, batch_first=True)

        self.reshape_size = int((256 *self.hidden_dim) ** 0.5)
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsample = nn.Upsample(size=(256, 224), mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(conv_channels, 1, kernel_size=1)  # Output layer



        # Detection head
        self.conv1_det = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2_det = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3_det = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsample_det = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv4_clshead = nn.Conv2d(conv_channels, 1, kernel_size=1)  # Output layer
        self.conv4_reghead = nn.Conv2d(conv_channels, 2, kernel_size=1)  # Output layer

    def forward(self, x):
       
        
        # x_t = x.view(x.shape[0], x.shape[1], -1) 
        # import pdb
        # pdb.set_trace()
        # x = (B, 32, 512, 256)
        x= x.permute(0, 2, 3, 1) # (1, 512, 256, 32) > (1, 256, 32, 512)
        # x = x.permute(0, 3, 1, 2)
        x_t = x.reshape(x.shape[0], x.shape[1], -1)
        # print("x_t_in", x_t.shape)
        for linear in self.linears:
            x_t = F.relu(linear(x_t))
            # print("x_t_Linear", x_t.shape)
        out, h = self.gru(x_t)
        # print("out", out.shape)
        lstm_concat = out
        x_t = lstm_concat.reshape(lstm_concat.shape[0], -1)
        x_flatten = x_t.view(-1,1, self.reshape_size, self.reshape_size)
        # print("x_t", x_t.shape)
        x_seg = F.relu(self.conv1(x_flatten))
        x_seg = F.relu(self.conv2(x_seg))
        x_seg = F.relu(self.conv3(x_seg))
        x_seg = self.upsample(x_seg)
        out_seg = self.conv4(x_seg)


        x_det = F.relu(self.conv1_det(x_flatten))
        x_det = F.relu(self.conv2_det(x_det))
        x_det = F.relu(self.conv3_det(x_det))
        x_det = self.upsample_det(x_det)
        x_det_cls = torch.sigmoid(self.conv4_clshead(x_det))
        x_det_reg = self.conv4_reghead(x_det)

        out_det = torch.cat([x_det_cls, x_det_reg], dim=1)

        out = {'Detection':[],'Segmentation':[]}

        out['Detection'] = out_det
        out['Segmentation'] = out_seg
        # x_t = x_t.view(x_t.shape[0], 1, 126, 224)
        return out



class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class ChirpNet2(nn.Module):
    def __init__(
        self,
        num_features=192,
        seq_len=64,
        num_layers=1,
        linear_dims=[32, 64],
        conv_channels=32,
        dropout_prob=0.1
    ):
        super(ChirpNet2, self).__init__()
        channels = 32  # as per your original code
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # ---------- Encoder (Linear + LayerNorm) ----------
        self.linears = nn.ModuleList()
        self.encode_norms = nn.ModuleList()
        last_dim = num_features

        self.input_normalization = nn.LayerNorm(last_dim * channels)

        for next_dim in linear_dims:
            self.linears.append(nn.Linear(last_dim * channels, next_dim * channels))
            # Add a LayerNorm for the new dimension:
            self.encode_norms.append(nn.LayerNorm(next_dim * channels))
            last_dim = next_dim

        self.dim = last_dim * channels
        self.seq_len = seq_len

        # ---------- Positional Encoding ----------
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, self.dim))

        # ---------- Transformer Layers (Independent) ----------
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(self.dim),
                "attn": SimpleSelfAttention(self.dim),
                "norm2": nn.LayerNorm(self.dim),
                "mlp": nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.GELU(),
                    nn.Dropout(p=self.dropout_prob)
                )
            }) for _ in range(num_layers)
        ])



        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsample = nn.Upsample(size=(256, 224), mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(conv_channels, 1, kernel_size=1)  # Output layer


        # Detection head
        self.conv1_det = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2_det = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3_det = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsample_det = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv4_clshead = nn.Conv2d(conv_channels, 1, kernel_size=1)  # Output layer
        self.conv4_reghead = nn.Conv2d(conv_channels, 2, kernel_size=1)  # Output layer

    def encode(self, x):
        """
        x shape: (batch_size, seq_len, 16, num_features) by default,
                 which is (batch_size, seq_len, channels, features).
        We'll reshape to (batch_size, seq_len, channels*features)
        and run through linear layers + LayerNorm.
        """
        x = x.permute(0, 2, 3, 1)
        x_t = x.reshape(x.shape[0], x.shape[1], -1)

        x_t = self.input_normalization(x_t)

        # x_t = x.view(x.shape[0], x.shape[1], -1)  # (B, S, 16*num_features)

        for linear, ln in zip(self.linears, self.encode_norms):
            x_t = linear(x_t)         # Apply linear
            x_t = F.relu(x_t)         # Non-linear activation
            x_t = ln(x_t)             # LayerNorm for stable scaling

        return x_t

    def sequence_model(self, x): # modified by ChatGPT
        """
        Adds positional encoding and passes x through each
        Transformer-like layer (independent attention blocks).
        x shape: (B, seq_len, dim).
        """
        x = x + self.positional_encoding[:, :x.shape[1], :]
        for layer in self.transformer_layers:
            # Self‐attention residual
            res = x
            x_norm = layer["norm1"](x)
            attn_out, _ = layer["attn"](x_norm)
            x = res + attn_out

            # Single‐block MLP residual
            res = x
            y = layer["norm2"](x)      # pre‐norm
            y = layer["mlp"](y)        # your nn.Sequential(Linear → GELU → Dropout)
            x = res + y
        return x

    def decode(self, x):
        """
        Reshape the sequence output into a 2D grid and run
        through convolutional decoder + upsampling.
        """
        batch_size, seq_len, feature_dim = x.size()
        height = int(math.sqrt(seq_len * feature_dim))
        width = height
        required_size = height * width

        # Flatten and slice if necessary
        x = x.view(batch_size, -1)[:, :required_size]
        x_flatten = x.view(batch_size, 1, height, width)


        x_seg = F.relu(self.conv1(x_flatten))
        x_seg = F.relu(self.conv2(x_seg))
        x_seg = F.relu(self.conv3(x_seg))
        x_seg = self.upsample(x_seg)
        out_seg = self.conv4(x_seg)


        x_det = F.relu(self.conv1_det(x_flatten))
        x_det = F.relu(self.conv2_det(x_det))
        x_det = F.relu(self.conv3_det(x_det))
        x_det = self.upsample_det(x_det)
        x_det_cls = torch.sigmoid(self.conv4_clshead(x_det))
        x_det_reg = self.conv4_reghead(x_det)

        out_det = torch.cat([x_det_cls, x_det_reg], dim=1)

        out = {'Detection':[],'Segmentation':[]}

        out['Detection'] = out_det
        out['Segmentation'] = out_seg
        # x_t = x_t.view(x_t.shape[0], 1, 126, 224)  # final shape
        return out

    def forward(self, x):
        x_encoded = self.encode(x)
        x_seq = self.sequence_model(x_encoded)
        output = self.decode(x_seq)
        return output

    def forward_w_enc_attn(self, x):
        """
        Same as forward, but also returns the encoded representation
        for additional analysis/debugging if needed.
        """
        x_encoded = self.encode(x)
        x_seq = self.sequence_model(x_encoded)
        # output = self.decode(x_seq)
        return  x_seq, x_encoded






# MambaSSM : VIM layers
# from .mamba_ssm import MambaSSMBlock


def causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, activation: str = "silu") -> torch.Tensor:
    """
    Causal 1D convolution.

    Args:
        x: Input tensor of shape (B, C, L).
        weight: Convolution weight of shape (C, 1, K).
        bias: Optional bias tensor of shape (C,).
        activation: Activation function to use; supported: "silu" or "relu".

    Returns:
        Tensor of shape (B, C, L) after convolution and activation.
    """
    B, C, L = x.shape
    K = weight.shape[-1]
    # Left-pad x with K-1 zeros.
    padding = (K - 1, 0)
    x_padded = F.pad(x, padding)
    y = F.conv1d(x_padded, weight, bias=bias, stride=1, padding=0, groups=C)
    if activation is None:
        return y
    elif activation.lower() == "silu":
        return F.silu(y)
    elif activation.lower() == "relu":
        return F.relu(y)
    else:
        raise ValueError("Unsupported activation: " + activation)

def causal_conv1d_update(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, activation: str = "silu") -> (torch.Tensor, torch.Tensor):
    """
    Causal convolution update for a time step.

    Args:
        x: Current input tensor of shape (B, C) (one time step).
        state: Buffer state tensor of shape (B, C, K-1) containing previous inputs.
        weight: Convolution weight of shape (C, 1, K).
        bias: Optional bias tensor of shape (C,).
        activation: Activation function ("silu" or "relu").

    Returns:
        y: Output tensor of shape (B, C) for the current time step.
        new_state: Updated state tensor of shape (B, C, K-1).
    """
    B, C = x.shape
    K = weight.shape[-1]
    x_new = x.unsqueeze(-1)  # (B, C, 1)
    full_input = torch.cat([state, x_new], dim=-1)  # (B, C, K)
    y = F.conv1d(full_input, weight, bias=bias, stride=1, padding=0, groups=C)
    if activation is None:
        y = y
    elif activation.lower() == "silu":
        y = F.silu(y)
    elif activation.lower() == "relu":
        y = F.relu(y)
    else:
        raise ValueError("Unsupported activation: " + activation)
    new_state = torch.cat([state[:, :, 1:], x_new], dim=-1)
    y = y.squeeze(-1)
    return y, new_state



class MambaSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None, bidirectional: bool = True,
                 d_conv: int = 4, expand: int = 2, use_fast_path: bool = True, init_layer_scale: float = None,
                 activation: str = "silu", layer_idx: int = None, device=None, dtype=None):
        """
        Mamba SSM block (__init__)

        Args:
            d_model: Model (feature) dimension.
            d_state: SSM state dimension.
            dt_rank: Rank for dt modulation (if None, set to ceil(d_model/16)).
            bidirectional: If True, process both forward and backward.
            d_conv: Convolution kernel width (for short conv pre-processing).
            expand: Expansion factor (the conv expands the channels to d_model * expand).
            use_fast_path: Whether to use fused fast-path (not implemented here; always uses reference path).
            init_layer_scale: Optional scaling factor for layer scaling.
            activation: Activation function ("silu" or "swish").
            device, dtype: Parameter Initialization.
        """
        super(MambaSSM, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank
        self.bidirectional = bidirectional
        self.use_fast_path = use_fast_path
        self.activation = activation
        self.layer_idx = layer_idx
        self.expand = expand
        self.init_layer_scale = init_layer_scale

        factory_kwargs = {"device": device, "dtype": dtype}

        # Project input to concatenated space: [dt_part, B_mod, C_mod]
        self.x_proj = nn.Linear(d_model * expand, self.dt_rank + 2 * d_state, bias=False, **factory_kwargs)

        # dt_proj projects the dt part to d_model (modulation vector).
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        target_dt = 0.01
        with torch.no_grad():
            # Inverse softplus initialization for dt_proj bias.
            self.dt_proj.bias.fill_(target_dt + math.log(1 - math.exp(-target_dt)))
        self.dt_proj.bias._no_reinit = True

        # A_log is stored in log-domain so that A = -exp(A_log) yields the decay factors.
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)\
                    .unsqueeze(0).expand(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.A_log._no_weight_decay = True

        # D is the learned skip scaling parameter.
        self.D = nn.Parameter(torch.ones(d_model, device=device))
        self.D._no_weight_decay = True

        # For bidirectional branch (we simply flip the sequence).
        self.bimamba_type = "none"

        # Save convolution parameters.
        self.d_conv = d_conv
        self.expand = expand

        # Define a causal convolution.
        self.conv1d = nn.Conv1d(
            in_channels=int(d_model * expand),
            out_channels=int(d_model * expand),
            kernel_size=d_conv,
            groups=int(d_model * expand),
            padding=d_conv - 1,  # Ensures causality
            bias=True,
            **factory_kwargs
        )
        # Choose activation for the convolution branch.
        if activation.lower() == "silu":
            self.act = F.silu
        elif activation.lower() in ["swish", "relu"]:
            self.act = F.silu  # For now using silu for both swish and relu.
        else:
            raise ValueError("Unsupported activation: " + activation)
        
        # Output projection: maps expanded dimension back to d_model.
        self.out_proj = nn.Linear(d_model * expand, d_model, bias=True, **factory_kwargs)
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones(d_model, device=device))

    def forward_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward recurrence (selective scan) over the sequence.

        Args:
            x: Input tensor of shape (B, L, d_model)
        Returns:
            y: Output tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        device = x.device

        # --- Step 1: Causal Convolution ---
        # Expand x to shape (B, L, d_model * expand) and then transpose to (B, d_model*expand, L)
        x_expanded = repeat(x, "b l d -> b l (d e)", e=self.expand)
        x_trans = x_expanded.transpose(1, 2)
        # Apply causal convolution using the imported function.
        x_conv = causal_conv1d_fn(x_trans, weight=self.conv1d.weight, bias=self.conv1d.bias, activation=self.activation)
        # Transpose back to (B, L, d_model*expand)
        x_conv = x_conv.transpose(1, 2)
        
        # --- Step 2: Projection ---
        # Project the convolved output into a concatenated vector:
        # [dt_part, B_mod, C_mod] where:
        #   dt_part: Used for computing modulation (via dt_proj)
        #   B_mod, C_mod: Used to modulate the recurrence state.
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt_part, B_mod, C_mod = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Compute dt using dt_proj followed by softplus.
        dt = F.softplus(self.dt_proj(dt_part))
        
        # --- Step 3: Recurrence ---
        # Initialize recurrence state h.
        h = torch.zeros(B, self.d_model, self.d_state, device=device, dtype=x.dtype)
        outputs = []
        
        A = -torch.exp(self.A_log)  # (d_model, d_state)

        for t in range(L):
            dt_t = dt[:, t, :]  # (B, d_model)
            # Compute decay: A = -exp(A_log)
            decay = torch.exp(dt_t.unsqueeze(-1) * A)  # (B, d_model, d_state)
            # Modulate the B_mod component.
            mod_B = dt_t.unsqueeze(-1) * B_mod[:, t, :].unsqueeze(1)  # (B, d_model, d_state)
            # Process the convolved output at time t:
            # Reshape x_conv[t] from (B, d_model*expand) to (B, d_model, expand) and average.
            x_conv_t = x_conv[:, t, :].view(B, self.d_model, self.expand).mean(dim=-1)
            # Update the state with decay and new contribution.
            h = h * decay + x_conv_t.unsqueeze(-1) * mod_B
            # Compute output y_t by summing the modulated state with a skip connection.
            y_t = torch.sum(h * C_mod[:, t, :].unsqueeze(1), dim=-1) + self.D * x_conv_t
            outputs.append(y_t)
        y_forward = torch.stack(outputs, dim=1)  # (B, L, d_model)
        return y_forward

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        Full forward pass that computes bidirectional recurrence.

        Args:
            hidden_states: Tensor of shape (B, L, d_model)
            inference_params: Optional cache for inference (unused here).
        Returns:
            Output tensor of shape (B, L, d_model)
        """
        y_fwd = self.forward_recurrence(hidden_states)
        if self.bidirectional:
            # Reverse the sequence for backward processing.
            y_bwd = self.forward_recurrence(hidden_states.flip(dims=[1])).flip(dims=[1])
            out = y_fwd + y_bwd
        else:
            out = y_fwd
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

    def step(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """
        Incremental processing for a single time step (useful for decoding).

        Args:
            hidden_states: (B, 1, d_model) representing the current token.
            conv_state: Buffer state for the convolution branch (B, d_model*expand, d_conv).
            ssm_state: Buffer state for the SSM branch (B, d_model*expand, d_state).
        Returns:
            out: Output tensor of shape (B, 1, d_model)
            Updated conv_state, ssm_state.
        """
        B, _, _ = hidden_states.shape
        x = hidden_states.squeeze(1)  # (B, d_model)
        # Expand x to match the conv branch dimensions.
        x_proj = self.x_proj(x.repeat(1, self.expand))
        dt_part, B_mod, C_mod = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_part))  # (B, d_model)
        A = -torch.exp(self.A_log)
        decay = torch.exp(dt.unsqueeze(-1) * A)
        mod_B = dt.unsqueeze(-1) * B_mod.unsqueeze(1)
        # Update the SSM state incrementally.
        ssm_state = ssm_state * decay + x.unsqueeze(-1) * mod_B
        y = torch.sum(ssm_state * C_mod.unsqueeze(1), dim=-1) + self.D * x
        y = self.out_proj(y)
        return y.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate cache for incremental inference.

        Returns:
            conv_state: Tensor of shape (B, d_model*expand, d_conv)
            ssm_state: Tensor of shape (B, d_model*expand, d_state)
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_conv,
                                 device=device, dtype=conv_dtype)
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_state,
                                device=device, dtype=ssm_dtype)
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        Retrieve or initialize inference cache from the given inference parameters.
        """
        if not hasattr(inference_params, "key_value_memory_dict"):
            inference_params.key_value_memory_dict = {}
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state, ssm_state = self.allocate_inference_cache(batch_size, None)
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class MambaSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None, bidirectional: bool = True,
                 d_conv: int = 4, expand: int = 2, **kwargs):
        """
        Convenience wrapper that applies the Mamba SSM block and then projects the output back to d_model.
        """
        super(MambaSSMBlock, self).__init__()
        # self.ssm = Mamba(d_model, d_state, dt_rank, bidirectional, d_conv=d_conv, expand=expand, **kwargs)
        self.ssm = Mamba(d_model, d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SSM block followed by an output projection.
        """
        y = self.ssm(x)
        return self.out_proj(y)



class ChirpNet3(nn.Module):
    def __init__(self, num_features=512, seq_len=256,
                 num_layers=1, linear_dims=[8, 8], conv_channels=16,
                 dropout_prob=0.3):
        super(ChirpNet3, self).__init__()
        channels = 32
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Encoder: Linear layers with normalization.
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        last_dim = num_features
        for next_dim in linear_dims:
            self.linears.append(nn.Linear(last_dim * channels, next_dim * channels))
            self.norms.append(nn.LayerNorm(next_dim * channels))
            last_dim = next_dim

        self.dim = last_dim * channels
        self.seq_len = seq_len

        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, self.dim))

        # Use MambaSSMBlock as VIM layers; stacking multiple layers helps learn deeper representations.
        self.vim_layers = nn.ModuleList([
            MambaSSMBlock(self.dim, d_state=32, dt_rank=None, bidirectional=True, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])

        # Decoder: Merge skip connections from encoder and multiple encoder stages if needed.
        # Here we simply merge the final encoder output (skip) with the output of the sequential model.
        self.decoder_linear = nn.Linear(self.dim * 2, self.dim)

        # Convolutional decoder: More layers can help refine spatial details.
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(conv_channels, 1, kernel_size=1)  # Output layer

    def encode(self, x):
        x = x.permute(0, 2, 3, 1)
        batch_size, seq_len, _, _ = x.size()
        # x_t = x.view(batch_size, seq_len, -1)
        x_t = x.reshape(batch_size, seq_len, -1)
        # For potential multi-scale fusion you might store intermediate features here.
        for linear, norm in zip(self.linears, self.norms):
            x_t = F.relu(linear(x_t))
            x_t = norm(x_t)
        # Save the final encoder output as the skip connection.
        skip_feature = x_t  
        return x_t, skip_feature

    def sequence_model(self, x):
        # Add positional encoding.
        x = x + self.positional_encoding[:, :x.shape[1], :]
        # Stack multiple VIM (SSM) layers.
        for layer in self.vim_layers:
            x = layer(x)
        return x

    def decode(self, x, skip):
        # Combine the sequence output with the skip connection.
        x = torch.cat([x, skip], dim=-1)  # Shape: (batch, seq_len, dim*2)
        x = F.relu(self.decoder_linear(x))  # Shape: (batch, seq_len, dim)
        
        batch_size, seq_len, feature_dim = x.size()
        # Reshape features into a 2D feature map.
        total_features = seq_len * feature_dim
        height = int(math.sqrt(total_features))
        width = height
        required_size = height * width

        x = x.view(batch_size, -1)[:, :required_size]
        x_t = x.view(batch_size, 1, height, width)

        x_t = F.relu(self.conv1(x_t))
        x_t = F.relu(self.conv2(x_t))
        x_t = F.relu(self.conv3(x_t))
        x_t = self.upsample(x_t)
        x_t = self.conv4(x_t)
        # x_t = x_t.view(x_t.shape[0], 1, 126, 224)
        return x_t

    def forward(self, x):
        x_encoded, skip = self.encode(x)
        x_seq = self.sequence_model(x_encoded)
        output = self.decode(x_seq, skip)
        return output


if __name__=='__main__':

    # Test case
    num_features = 512
    hidden_dim = 1024
    num_layers = 1
    linear_dims = [64, 128]
    conv_channels = 32
    batch_size = 1


    # Initialize the model
    # model = ChirpNet1(512, 1024, 1, [64, 128], 32).to('cuda')

    # model = ChirpNet2(
    #     num_features=512, 
    #     seq_len=256, 
    #     num_layers=1, 
    #     linear_dims=[32], 
    #     conv_channels=32,
    #     dropout_prob=0.1
    # ).to('cuda')

    model = ChirpNet3(
    num_features=512,
    seq_len=256,
    num_layers=1,
    linear_dims=[64, 32],
    conv_channels=32,
    dropout_prob=0.3
    ).to('cuda')

    tensor = torch.rand((1, 512, 256, 32)).to('cuda')
    output = model(tensor)

    print(output["Segmentation"].shape, output["Detection"].shape)