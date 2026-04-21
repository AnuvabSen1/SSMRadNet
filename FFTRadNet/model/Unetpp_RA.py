import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np

from ptflops import get_model_complexity_info


class ADCtoRAMapFull(nn.Module):
    """
    Input:
      x: (B, 512, 256, 32) where last dim is [real(16), imag(16)]

    Output:
      reduce=False -> (B, 256, range(=512), azimuth(A))
      reduce=True  -> (B, 1,   range(=512), azimuth(A))  (sum over doppler bins)

    Notes:
      - Uses the same DDM demux indexing and AoA beamforming as your RadarSignalProcessing.__get_RA()
      - Uses torch.fft.fft for range and doppler FFTs
      - Uses AoA_mat['Signal'][..., elev_index] and AoA_mat['H'][0] from the calibration dict
    """

    def __init__(
        self,
        path_calib_mat: str = "../SignalProcessing/CalibrationTable.npy",
        num_samples: int = 512,
        num_chirps: int = 256,
        num_rx: int = 16,
        num_reduced_doppler: int = 16,
        num_chirps_per_loop: int = 16,
        elev_index: int = 5,
        tx_drop_first_k: int = 5,  # matches: seq = [seq[0]] + seq[5:]
    ):
        super().__init__()

        self.num_samples = int(num_samples)
        self.num_chirps = int(num_chirps)
        self.num_rx = int(num_rx)

        # ---- load calibration dict (same structure as your snippet) ----
        aoa = np.load(path_calib_mat, allow_pickle=True).item()

        calib = aoa["Signal"][..., elev_index]  # (A, V) complex (typically)
        window = aoa["H"][0]                    # (V,) real (typically)

        # Convert calib to complex64 torch tensor
        calib_t = torch.as_tensor(calib)
        if not torch.is_complex(calib_t):
            calib_t = calib_t.to(torch.complex64)
        else:
            calib_t = calib_t.to(torch.complex64)

        window_t = torch.as_tensor(window, dtype=torch.float32)

        # Register as buffers so .to(device) moves them with the module
        self.register_buffer("calib_mat", calib_t, persistent=False)  # (A, V)
        self.register_buffer("bf_window", window_t, persistent=False) # (V,)

        # ---- build range/doppler windows (same 0.54-0.46cos formula) ----
        r = torch.arange(self.num_samples, dtype=torch.float32)
        d = torch.arange(self.num_chirps, dtype=torch.float32)

        range_win = 0.54 - 0.46 * torch.cos((2.0 * math.pi * r) / (self.num_samples - 1))
        doppler_win = 0.54 - 0.46 * torch.cos((2.0 * math.pi * d) / (self.num_chirps - 1))

        self.register_buffer("range_win", range_win, persistent=False)     # (R,)
        self.register_buffer("doppler_win", doppler_win, persistent=False) # (D,)

        # ---- precompute DDM demux doppler indices (D, Tx=12) ----
        dividend_constant_arr = np.arange(
            0, num_reduced_doppler * num_chirps_per_loop, num_reduced_doppler
        )  # [0,16,32,...,240] (len=16)

        doppler_idx = []
        for doppler_bin in range(self.num_chirps):
            seq = np.remainder(doppler_bin + dividend_constant_arr, self.num_chirps)  # (16,)
            # matches your snippet: keep seq[0] and seq[5:]
            seq = np.concatenate([seq[:1], seq[tx_drop_first_k:]], axis=0)            # (12,)
            doppler_idx.append(seq)

        doppler_idx = np.stack(doppler_idx, axis=0)  # (D=256, Tx=12)
        doppler_idx_t = torch.as_tensor(doppler_idx, dtype=torch.long)
        self.register_buffer("doppler_idx", doppler_idx_t, persistent=False)

        # Sanity: virtual array length must match calibration/window
        self.num_tx = int(doppler_idx_t.shape[1])     # 12
        self.vdim = self.num_tx * self.num_rx         # V = 12*16=192 expected

        if self.bf_window.numel() != self.vdim:
            raise ValueError(
                f"Beamforming window length mismatch: window={self.bf_window.numel()} "
                f"but expected V={self.vdim} (=Tx{self.num_tx}*Rx{self.num_rx})."
            )
        if self.calib_mat.shape[1] != self.vdim:
            raise ValueError(
                f"Calibration matrix shape mismatch: calib_mat.shape={tuple(self.calib_mat.shape)} "
                f"but expected second dim V={self.vdim}."
            )

    def forward(self, x: torch.Tensor, reduce: bool = False) -> torch.Tensor:
        """
        x: (B, 512, 256, 32) float tensor
        reduce: if True, sums over doppler bins -> (B,1,R,A)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim==4, got {x.ndim} with shape {tuple(x.shape)}")
        B, R, D, C = x.shape
        if (R, D, C) != (self.num_samples, self.num_chirps, 2 * self.num_rx):
            raise ValueError(
                f"Expected x shape (B,{self.num_samples},{self.num_chirps},{2*self.num_rx}), "
                f"got {tuple(x.shape)}"
            )

        # ---- real/imag -> complex: (B, R, D, Rx) ----
        real = x[..., : self.num_rx].to(torch.float32)
        imag = x[..., self.num_rx :].to(torch.float32)
        adc = torch.complex(real, imag)  # complex64 (components float32)

        # ---- remove DC offset over (range, chirp) per RX channel ----
        adc = adc - adc.mean(dim=(1, 2), keepdim=True)

        # ---- Range FFT (dim=1) with range window ----
        adc = adc * self.range_win.view(1, R, 1, 1)
        rng_fft = torch.fft.fft(adc, n=R, dim=1)

        # ---- Doppler FFT (dim=2) with doppler window ----
        rng_fft = rng_fft * self.doppler_win.view(1, 1, D, 1)
        rd = torch.fft.fft(rng_fft, n=D, dim=2)  # (B, R, D, Rx)

        # ---- DDM demux: build virtual array spectrum per (range, doppler) ----
        # We want: demux -> (B, R, D, Tx, Rx)
        rd_perm = rd.permute(0, 1, 3, 2).contiguous()   # (B, R, Rx, D)

        # Build index: (B, R, Rx, D, Tx)
        idx = self.doppler_idx.view(1, 1, 1, D, self.num_tx).expand(B, R, self.num_rx, D, self.num_tx)

        # Expand input to match idx shape in the last dim: (B, R, Rx, D, Tx)
        rd_rep = rd_perm.unsqueeze(-1).expand(B, R, self.num_rx, D, self.num_tx)

        # Gather along doppler dim (dim=3) -> (B, R, Rx, D, Tx)
        demux = rd_rep.gather(dim=3, index=idx)

        # Now to (B, R, D, Tx, Rx)
        demux = demux.permute(0, 1, 3, 4, 2).contiguous()

        # Flatten virtual array: V = Tx*Rx
        mimo = demux.reshape(B, R * D, self.vdim)       # (B, N=R*D, V)

        # Window over virtual array to reduce sidelobes
        mimo = mimo * self.bf_window.view(1, 1, self.vdim)  # broadcast (B,N,V)

        # Prepare for beamforming matmul: (B, V, N)
        mimo_t = mimo.transpose(1, 2).contiguous()      # (B, V, N)

        # Beamforming: (A, V) @ (B, V, N) -> (B, A, N)
        az = torch.matmul(self.calib_mat, mimo_t)       # complex
        az = torch.abs(az)                              # (B, A, N) real

        # Reshape to (B, D, R, A) to match your required output
        A = az.shape[1]
        az = az.view(B, A, R, D)                        # (B, A, R, D)
        ra = az.permute(0, 3, 2, 1).contiguous()        # (B, D, R, A)

        if reduce:
            ra = ra.sum(dim=1, keepdim=True)            # (B, 1, R, A)

        return ra


class ADCtoRAMap(nn.Module):
    """
    ADC (B, 512, C, 32) -> RA (B, 1, 512, numAz)

    Input format assumption:
      - adc.shape = (B, S=512, C=256, 32)
      - adc[..., :16] are REAL parts for 16 RX channels
      - adc[..., 16:] are IMAG parts for 16 RX channels
      - So complex RX tensor is (B, 512, C, 16)

    Processing matches your CUDA snippet:
      - Range FFT along samples
      - Doppler FFT along selected chirps_used
      - DDM demux by selecting Doppler bins (N x 12) per Doppler index
      - Virtual window and beamforming using calib matrix
      - Sum over Doppler to get RA
    """
    def __init__(
        self,
        path_calib_mat: str="../SignalProcessing/CalibrationTable.npy",
        chirps_used: int = 16,
        elev_idx: int = 5,
    ):
        super().__init__()

        # ---- Radar params (same as your snippet) ---- :contentReference[oaicite:1]{index=1}
        self.numSamplePerChirp = 512
        self.numRxAnt = 16
        self.numTxAnt = 12
        self.numChirpsPerLoop = 16  # DDM code length L :contentReference[oaicite:2]{index=2}

        self.N = int(chirps_used)
        assert self.N > 0
        assert self.N % self.numChirpsPerLoop == 0, (
            f"chirps_used={self.N} must be a multiple of {self.numChirpsPerLoop} for bin-aligned DDM demux."
        )

        # ---- Load calibration (CPU load; buffers move with .to(device)) ---- :contentReference[oaicite:3]{index=3}
        aoa = np.load(path_calib_mat, allow_pickle=True).item()
        calib = aoa["Signal"][..., elev_idx]  # (numAz, numVirtual) complex
        win = aoa["H"][0]                    # (numVirtual,)

        CalibMat = torch.from_numpy(calib).to(torch.complex64)   # (numAz, numVirtual)
        window = torch.from_numpy(win).to(torch.float32)         # (numVirtual,)

        self.numAz = int(aoa["Signal"].shape[0])
        self.numVirtual = int(CalibMat.shape[1])

        # Sanity: expected virtual channels = 12 * 16 in your snippet flow :contentReference[oaicite:4]{index=4}
        expected_virtual = self.numTxAnt * self.numRxAnt
        assert self.numVirtual == expected_virtual, (
            f"Calibration expects numVirtual={self.numVirtual}, but demux builds {expected_virtual} (=12*16)."
        )

        self.register_buffer("CalibMat", CalibMat, persistent=True)
        self.register_buffer("virtual_window", window, persistent=True)

        # ---- Precompute Hamming windows (range & doppler) ---- :contentReference[oaicite:5]{index=5}
        n_r = torch.arange(self.numSamplePerChirp, dtype=torch.float32)
        w_r = 0.54 - 0.46 * torch.cos((2 * math.pi * n_r) / (self.numSamplePerChirp - 1))

        n_d = torch.arange(self.N, dtype=torch.float32)
        w_d = 0.54 - 0.46 * torch.cos((2 * math.pi * n_d) / (self.N - 1))

        self.register_buffer("range_win", w_r, persistent=True)    # (S,)
        self.register_buffer("doppler_win", w_d, persistent=True)  # (N,)

        # ---- Precompute DDM demux Doppler indices (N,12) ---- :contentReference[oaicite:6]{index=6}
        ddm_stride = self.N // self.numChirpsPerLoop  # Δ
        dividend = torch.arange(0, ddm_stride * self.numChirpsPerLoop, ddm_stride, dtype=torch.long)  # (16,)
        base = torch.arange(self.N, dtype=torch.long).view(self.N, 1)  # (N,1)
        seq = (base + dividend.view(1, -1)) % self.N                   # (N,16)

        keep = torch.tensor([0] + list(range(5, 16)), dtype=torch.long)  # (12,)
        doppler_idx = seq.index_select(dim=1, index=keep)                # (N,12)

        self.register_buffer("doppler_idx", doppler_idx, persistent=True)

    def forward(self, adc: torch.Tensor, group_index: int = 0, reduce: bool = False) -> torch.Tensor:
        """
        adc: (B, 512, C, 32) real tensor (float/half/int ok)
        group_index: selects chirp block [group_index*N : (group_index+1)*N]
                     default 0 => first 16 chirps (when N=16)

        returns:
          ra: (B, 1, 512, numAz) float32
        """
        assert adc.ndim == 4 and adc.shape[1] == self.numSamplePerChirp and adc.shape[-1] == 32, (
            f"Expected adc shape (B, 512, C, 32), got {tuple(adc.shape)}"
        )
        B, S, C, _ = adc.shape
        start = group_index * self.N
        end = start + self.N
        if end > C:
            raise ValueError(f"group_index={group_index} with chirps_used={self.N} exceeds available chirps C={C}.")

        # Split real/imag for 16 RX channels -> complex (B, S, C, 16)
        real = adc[..., :16].to(torch.float32)
        imag = adc[..., 16:].to(torch.float32)
        x_full = torch.complex(real, imag)  # complex64

        # Select chirp block -> (B, S, N, 16)
        x = x_full[:, :, start:end, :]

        # Remove per-RX mean over (range, chirps) per batch (matches your intent) :contentReference[oaicite:7]{index=7}
        x = x - x.mean(dim=(1, 2), keepdim=True)

        # Range FFT (dim=1) :contentReference[oaicite:8]{index=8}
        x = x * self.range_win.view(1, S, 1, 1)
        range_fft = torch.fft.fft(x, n=S, dim=1)

        # Doppler FFT (dim=2) over ONLY N=chirps_used :contentReference[oaicite:9]{index=9}
        range_fft = range_fft * self.doppler_win.view(1, 1, self.N, 1)
        RD = torch.fft.fft(range_fft, n=self.N, dim=2)  # (B, S, N, 16)

        # ---- DDM demux: select Doppler bins -> (B, S, N, 12, 16) ---- :contentReference[oaicite:10]{index=10}
        # Use gather (compile-friendly) to emulate RD[:, doppler_idx, :]
        # RD_perm: (B, S, 16, N)
        RD_perm = RD.permute(0, 1, 3, 2)

        # Expand to (B, S, 16, N, 12)
        RD_exp = RD_perm.unsqueeze(-1).expand(B, S, self.numRxAnt, self.N, self.numTxAnt)

        # idx: (B, S, 16, N, 12) with values in [0, N)
        idx = self.doppler_idx.view(1, 1, 1, self.N, self.numTxAnt).expand(B, S, self.numRxAnt, self.N, self.numTxAnt)

        sel = torch.gather(RD_exp, dim=3, index=idx)  # (B, S, 16, N, 12)
        mimo = sel.permute(0, 1, 3, 4, 2)            # (B, S, N, 12, 16)

        # Flatten virtual array: (B, S*N, 12*16) :contentReference[oaicite:11]{index=11}
        MIMO = mimo.reshape(B, S * self.N, self.numTxAnt * self.numRxAnt)

        # Apply virtual window
        MIMO = MIMO * self.virtual_window.view(1, 1, -1)

        # Beamforming: (numAz, numVirtual) @ (B, numVirtual, S*N) -> (B, numAz, S*N) :contentReference[oaicite:12]{index=12}
        Az = torch.abs(torch.matmul(self.CalibMat, MIMO.transpose(1, 2)))

        # Reshape to (B, numAz, S, N), Doppler-collapse, then (B, 1, S, numAz) :contentReference[oaicite:13]{index=13}
        Az = Az.view(B, self.numAz, S, self.N)
        RA = Az.transpose(1, 3)             # (B, S, numAz)
        
        if reduce:
            RA = RA.sum(dim=1, keepdim=True)            # (B, 1, R, A)

        return RA
    
# --------------------------
# Small utilities
# --------------------------
def _best_gn_groups(c: int, max_groups: int = 8) -> int:
    g = min(max_groups, c)
    while g > 1:
        if c % g == 0:
            return g
        g -= 1
    return 1


def make_norm2d(norm: str, c: int) -> nn.Module:
    norm = (norm or "gn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm == "gn":
        return nn.GroupNorm(_best_gn_groups(c, 8), c)
    if norm == "in":
        return nn.InstanceNorm2d(c, affine=True)
    return nn.Identity()


def upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # bilinear upsample to match ref spatial size
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------
# Basic building block
# --------------------------
class BasicBlock(nn.Module):
    """
    Residual 2x(Conv3x3 + Norm + ReLU) with optional 1x1 skip when channels change.
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm2d(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm2d(norm, out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                make_norm2d(norm, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        out = self.act(out + identity)
        return out


# --------------------------
# UNet++ for Radar
#  - Input:  B x 16 x 256 x 448
#  - Stem compresses width by 2: -> B x C0 x 256 x 224
#  - Encoder downsamples HEIGHT only (2,1): keeps width=224
#  - Segmentation logits: B x 1 x 256 x 224  (from x0_4)
#  - Detection:          B x 3 x 128 x 224  (from x1_3)
# --------------------------
class UNetPlusPlusRadar(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        filters=(64, 128, 256, 512, 512),
        norm: str = "gn",
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(filters) == 5, "filters must have 5 levels (C0..C4)."
        self.filters = list(filters)
        C0, C1, C2, C3, C4 = self.filters

        # Width-compression stem: 448 -> 224 (stride on W only)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C0, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            make_norm2d(norm, C0),
            nn.ReLU(inplace=True),
        )

        # Encoder blocks (x0_0..x4_0)
        self.enc0 = BasicBlock(C0, C0, norm=norm, dropout=dropout)
        self.pool0 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/2, W same
        self.enc1 = BasicBlock(C0, C1, norm=norm, dropout=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.enc2 = BasicBlock(C1, C2, norm=norm, dropout=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.enc3 = BasicBlock(C2, C3, norm=norm, dropout=dropout)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.enc4 = BasicBlock(C3, C4, norm=norm, dropout=dropout)

        # UNet++ nested decoder blocks: x{i}_{j} for i=0..(4-j), j=1..4
        # x{i}_{j} = Block( cat( x{i}_{0..j-1}, up(x{i+1}_{j-1}) ) ) -> channels = filters[i]
        self.dec = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                in_ch = self.filters[i] * j + self.filters[i + 1]
                out_ch = self.filters[i]
                self.dec[f"x{i}_{j}"] = BasicBlock(in_ch, out_ch, norm=norm, dropout=dropout)

        # Detection head at (H/2, W) = (128, 224): use x1_3 (best-refined at level 1)
        self.det_neck = nn.Sequential(
            nn.Conv2d(C1, 256, kernel_size=1, bias=False),
            make_norm2d(norm, 256),
            nn.ReLU(inplace=True),
            BasicBlock(256, 256, norm=norm, dropout=dropout),
        )
        self.clshead = nn.Conv2d(256, 1, kernel_size=1)  # sigmoid
        self.reghead = nn.Conv2d(256, 2, kernel_size=1)  # linear
        # Detection output: cat([sigmoid(cls), reg]) => B x 3 x 128 x 224

        # Segmentation head at (H, W) = (256, 224): from x0_4
        # Kept "roughly like" your snippet but driven by UNet++ output
        self.freespace = nn.Sequential(
            nn.Conv2d(C0, 256, kernel_size=1, bias=False),
            make_norm2d(norm, 256),
            nn.ReLU(inplace=True),
            BasicBlock(256, 128, norm=norm, dropout=dropout),
            BasicBlock(128, 64, norm=norm, dropout=dropout),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 16, 256, 448)
        returns:
          out['Detection']    : (B, 3, 128, 224)
          out['Segmentation'] : (B, 1, 256, 224)  logits
        """
        # --- Encoder ---
        x0_0 = self.enc0(self.stem(x))            # (B, C0, 256, 224)
        x1_0 = self.enc1(self.pool0(x0_0))        # (B, C1, 128, 224)
        x2_0 = self.enc2(self.pool1(x1_0))        # (B, C2,  64, 224)
        x3_0 = self.enc3(self.pool2(x2_0))        # (B, C3,  32, 224)
        x4_0 = self.enc4(self.pool3(x3_0))        # (B, C4,  16, 224)

        nodes = {}
        nodes[(0, 0)] = x0_0
        nodes[(1, 0)] = x1_0
        nodes[(2, 0)] = x2_0
        nodes[(3, 0)] = x3_0
        nodes[(4, 0)] = x4_0

        # --- UNet++ nested decoder ---
        for j in range(1, 5):
            for i in range(0, 5 - j):
                ref = nodes[(i, 0)]  # target spatial size for this level
                up = upsample_to(nodes[(i + 1, j - 1)], ref)

                cat_list = [nodes[(i, k)] for k in range(0, j)] + [up]
                x_ij = torch.cat(cat_list, dim=1)

                nodes[(i, j)] = self.dec[f"x{i}_{j}"](x_ij)

        # --- Outputs ---
        # Segmentation: (B, 1, 256, 224)
        seg_logits = self.freespace(nodes[(0, 4)])

        # Detection: (B, 3, 128, 224)
        det_feat = self.det_neck(nodes[(1, 3)])
        cls = torch.sigmoid(self.clshead(det_feat))
        reg = self.reghead(det_feat)
        det = torch.cat([cls, reg], dim=1)

        return {
            "Detection": det,
            "Segmentation": seg_logits,
        }


class RaTok(nn.Module):
    """
    Output format:
        cls = sigmoid(clshead(x))  -> (B,1,128,224)
        reg = reghead(x)           -> (B,2,128,224)
        out = cat([cls, reg], dim=1) -> (B,3,128,224)
    """
    def __init__(
        self,
        beamformer_chunkwise = True,
        beamformer_all_chunks = False,
        beamformer_reduce = False
    ):
        super().__init__()
        self.beamformer_chunkwise = beamformer_chunkwise
        self.beamformer_all_chunks = beamformer_all_chunks
        self.beamformer_reduce = beamformer_reduce

        self.RA_channels = 256

        if self.beamformer_chunkwise == False:
            if self.beamformer_reduce == False:
                self.RA_channels = 256
            else:
                self.RA_channels = 1
        else:
            if self.beamformer_all_chunks:
                if self.beamformer_reduce == False:
                    self.RA_channels = 256
                else:
                    self.RA_channels = 256//16
            else:
                if self.beamformer_reduce == False:
                    self.RA_channels = 256//16
                else:
                    self.RA_channels = 1


        if self.beamformer_chunkwise:
            self.beamformer = ADCtoRAMap()
        else:
            self.beamformer = ADCtoRAMapFull()


        self.unet = UNetPlusPlusRadar(in_channels=self.RA_channels, filters=(24, 48, 96, 192, 192))


    def forward(self, adc: torch.Tensor) -> torch.Tensor:
        # radar_ra: (B, 1, 512, 751)

        with torch.no_grad():

            if self.beamformer_chunkwise == False:
                if self.beamformer_reduce == False:
                    radar_ra = self.beamformer(adc, reduce = False)
                else:
                    radar_ra = self.beamformer(adc, reduce=True)
            else:
                if self.beamformer_all_chunks:
                    if self.beamformer_reduce == False:
                        radar_ra = [self.beamformer(adc, group_index=chunk_no, reduce=False) for chunk_no in range(16)]  # 16 x (B,1,R,A)
                        radar_ra = torch.cat(radar_ra, dim=1)  # (B,16,R,A) because dim=1 concatenates the 1-channels
                    else:
                        radar_ra = [self.beamformer(adc, group_index=chunk_no, reduce=True) for chunk_no in range(16)]  # 16 x (B,1,R,A)
                        radar_ra = torch.cat(radar_ra, dim=1)  # (B,16,R,A) because dim=1 concatenates the 1-channels
                else:
                    if self.beamformer_reduce == False:
                        radar_ra = self.beamformer(adc, reduce = False)
                    else:
                        radar_ra = self.beamformer(adc, reduce = True)

            radar_ra = F.interpolate(radar_ra, (256, 448))

        out = self.unet(radar_ra)

        return out


if __name__ == "__main__":

    model = RaTok(unet_input_channel=256, chirp_style='full').cuda()
    x = torch.rand(2, 512, 256, 32).cuda()
    with torch.no_grad():
        y = model(x)
        print(y['Detection'].shape, y['Segmentation'].shape)

    
    macs_str, params_str = get_model_complexity_info(
        model,
        input_res=(512, 256, 32),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"MACs:   {macs_str}")
    print(f"Params: {params_str}")


    import time

    # ---- setup ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, R, C, A = 2, 512, 256, 32  # (batch, range, chirps, adc?)
    warmup = 30
    iters  = 100

    # ---- warmup ----
    with torch.inference_mode():
        for _ in range(warmup):
            x = torch.rand(B, R, C, A, device=device)
            _ = model(x)

    # ---- benchmark ----
    times_ms = []

    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

            for i in range(iters):
                x = torch.rand(B, R, C, A, device=device)

                starts[i].record()
                _ = model(x)
                ends[i].record()

            torch.cuda.synchronize()
            times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]

        else:
            for _ in range(iters):
                x = torch.rand(B, R, C, A, device=device)
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)

    mean_ms = times_ms.mean()
    p95_ms  = np.percentile(times_ms, 95)

    samples_per_sec = B / (mean_ms / 1000.0)
    batches_per_sec = 1.0 / (mean_ms / 1000.0)

    print(f"Device: {device}")
    print(f"Batch size: {B}, iters: {iters}")
    print(f"Mean latency: {mean_ms:.3f} ms | P95: {p95_ms:.3f} ms")
    print(f"Throughput: {samples_per_sec:.2f} samples/s  ({batches_per_sec:.2f} batches/s)")