import torch

class Diffusion:
    def __init__(self, timesteps=100, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        beta_start = 1e-4
        beta_end = 0.02

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # 🔥 提前算好
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t].view(-1,1,1,1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1,1,1,1)

        return sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise