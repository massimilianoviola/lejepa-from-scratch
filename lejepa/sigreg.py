import torch


class SIGReg(torch.nn.Module):
    """Penalizes divergence of embeddings from an isotropic Gaussian.
    Projects them onto random 1-D directions and applies the Epps-Pulley test to
    each slice: T = N * integral |ecf(t) - phi(t)|^2 w(t) dt, comparing the empirical
    characteristic function of the projections against that of N(0, 1). Here we also
    leverage the symmetric property of the ECF/CF improve the quadrature efficiency.
    We integrate on [0, t_max] and double, instead of integrating on [-t_max, t_max].
    """

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256
    ) -> None:
        super().__init__()
        self.num_slices = num_slices  # Number of 1D projections
        t = torch.linspace(0, t_max, knots)  # Interval where the two CFs are compared
        dt = t_max / (knots - 1)
        # We only integrate over [0, t_max] as the integrand is even, so double the
        # interior weights to recover the full integral
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        # Phi serves as both the CF of N(0, 1) and the Epps-Pulley window w(t)
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        assert proj.ndim == 3, f"expected (views, batch, dims), got {tuple(proj.shape)}"
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))  # Unit norm so slices of N(0, I) are N(0, 1)
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (views, batch, slices, knots)
        # Squared modulus of ecf(t) - phi(t), split into real and imaginary parts
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
