import math
import torch
from transformers import TrainerCallback

class psp_callback(TrainerCallback):
    def __init__(self, mask_fraction=0.5, max_grad_norm=1.0, eps=1e-6):
        self.mask_fraction = mask_fraction
        self.max_grad_norm = max_grad_norm
        self.eps = eps

    def on_before_optimizer_step(self, args, state, control, optimizer, **kwargs):
        if len(optimizer.param_groups) == 0:
            return
        device = next(iter(optimizer.param_groups[0]['params'])).device
        param_info = []
        total_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    g_norm_sq = param.grad.float().pow(2).sum()
                    p_norm_sq = param.detach().float().pow(2).sum()
                    total_norm_sq += g_norm_sq
                    param_info.append((p_norm_sq, g_norm_sq, param))
        if not param_info:
            return
        param_info.sort(key=lambda x: x[0].item(), reverse=True)
        num_to_mask = max(1, min(int(len(param_info) * self.mask_fraction), len(param_info) - 1))
        masked_norm_sum = torch.zeros((), device=device, dtype=torch.float32)
        for i in range(num_to_mask):
            _, gns, param = param_info[i]
            param.grad.zero_()
            masked_norm_sum += gns
        total_norm_sq -= masked_norm_sum
        total_norm = torch.sqrt(total_norm_sq).item()
        if not math.isfinite(total_norm):
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.zero_()
            return
        clip_coef = self.max_grad_norm / (total_norm + self.eps)
        if clip_coef < 1.0:
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.mul_(clip_coef)
