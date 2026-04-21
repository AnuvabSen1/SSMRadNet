class ComplexRadarMixer(nn.Module):
    """
    Input:  x  (B?, 512, 256, 16)  complex64/complex128
    Params: w256   (12, 256)       learnable complex (12 1D signals across 256)
            w2d    (512, 256)      learnable complex (single 2D signal)
    Output: y  (B?, 512, 256, 192) = (16 * 12)
    """
    def __init__(self, n_signals_along_256: int = 12, dtype=torch.cfloat, device=None):
        super().__init__()
        assert dtype in (torch.cfloat, torch.cdouble), "Use a complex dtype."

        self.n_sig = n_signals_along_256
        self.dtype = dtype

        # --- Initialize 12 complex 1D signals over length 256 ---
        # Initialize with unit-magnitude complex phases + small noise for stability
        phase_1d = torch.rand(self.n_sig, 256, device=device) * 2 * math.pi
        w256_init = torch.cos(phase_1d) + 1j * torch.sin(phase_1d)
        w256_init = w256_init.to(dtype)

        # --- Initialize single complex 2D signal over (512, 256) ---
        phase_2d = torch.rand(512, 256, device=device) * 2 * math.pi
        w2d_init = torch.cos(phase_2d) + 1j * torch.sin(phase_2d)
        w2d_init = w2d_init.to(dtype)

        self.w256 = nn.Parameter(w256_init)   # (12, 256)
        self.w2d  = nn.Parameter(w2d_init)    # (512, 256)

        self.w256_norm = nn.LayerNorm (16*12*2)
        self.w2d_norm = nn.LayerNorm(16*12*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 512, 256, 16) complex tensor
        returns: (..., 512, 256, 192)
        """
        x = torch.complex(x[:, :, :, :16], x[:, :, :, 16:])
        if not torch.is_complex(x):
            raise TypeError("Input x must be a complex tensor (torch.cfloat/cdouble).")
        if x.shape[-3:] != (512, 256, 16):
            raise ValueError(f"Expected last dims (512, 256, 16), got {x.shape[-3:]}")

        B_prefix = x.shape[:-3]  # allow optional batch or leading dims
        # Step 1: multiply by 12 complex signals along the 256-dim and expand channels
        # Broadcast w256: (12,256) -> (1,...,1, 1, 256, 1, 12)
        w256_b = self.w256.view(*([1]*len(B_prefix)), 1, 256, 1, self.n_sig)
        # x: (..., 512, 256, 16) -> (..., 512, 256, 16, 1)
        x_exp = x.unsqueeze(-1)
        y = x_exp * w256_b  # (..., 512, 256, 16, 12)
        # Merge (16, 12) -> 192
        y = y.reshape(*B_prefix, 512, 256, 16 * self.n_sig)  # (..., 512, 256, 192)
        
        
        y = torch.cat([y.real, y.imag], dim=-1)
        y = self.w256_norm(y)
        y = torch.complex(y[:, :, :, :192], y[:, :, :, 192:])

        # Step 2: multiply by single complex 2D signal over (512,256)
        # w2d: (512,256) -> (1,...,1, 512, 256, 1)
        w2d_b = self.w2d.view(*([1]*len(B_prefix)), 512, 256, 1)
        y = y * w2d_b  # (..., 512, 256, 192)
        
        y = torch.cat([y.real, y.imag], dim=-1)
        y = self.w2d_norm(y)

        return y