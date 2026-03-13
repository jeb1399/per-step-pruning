import torch
import torch.distributed as dist
from transformers import TrainerCallback

class psp_callback(TrainerCallback):
    def __init__(self, mask_fraction=0.5, max_grad_norm=1.0, eps=1e-6):
        self.mask_fraction = mask_fraction
        self.masked_layers = []
        self.max_grad_norm = max_grad_norm
        self.eps = eps

    # Try to fix "nan" gradients and 0.0 loss
    def on_after_backward(self, args, state, control, model, **kwargs):
        device = next(model.parameters()).device
        total_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        for param in model.parameters():
            if param.grad is not None:
                total_norm_sq += param.grad.data.float().pow(2).sum()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)

        total_norm = torch.sqrt(total_norm_sq.item())
        if not torch.isfinite(torch.tensor(total_norm)):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
            return
        clip_coef = self.max_grad_norm / (total_norm + self.eps)
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

    def on_step_begin(self, args, state, control, model, **kwargs):
        layer_scores = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                score = param.detach().float().norm().item()
                layer_scores.append((name, score))
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        num_to_mask = max(min(int(len(layer_scores) * self.mask_fraction), len(layer_scores) - 1), 1)
        # Better temporary parameter "pruning"
        if num_to_mask >= len(layer_scores):
            num_to_mask = max(0, len(layer_scores) - 1)
        self.masked_layers = [name for name, _ in layer_scores[:num_to_mask]]
        for name, param in model.named_parameters():
            if name in self.masked_layers:
                if not hasattr(param, '_psp_hook'):
                    param._psp_hook = param.register_hook(lambda grad, _=param: grad * 0.0)

    def on_step_end(self, args, state, control, model, **kwargs):
        for name, param in model.named_parameters():
            if hasattr(param, '_psp_hook'):
                param._psp_hook.remove()
                delattr(param, '_psp_hook')
