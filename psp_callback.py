import torch
from transformers import TrainerCallback

class psp_callback(TrainerCallback):
    def __init__(self, mask_fraction=0.5):
        self.mask_fraction = mask_fraction
        self.original_requires_grad = {}

    def on_step_begin(self, args, state, control, model, **kwargs):
        layer_scores = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                score = torch.norm(param).item()
                layer_scores.append((name, score))
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        num_to_mask = int(len(layer_scores) * self.mask_fraction)
        print(num_to_mask)
        if num_to_mask >= len(layer_scores):
            num_to_mask = max(0, len(layer_scores) - 1)
        self.masked_layers = [name for name, _ in layer_scores[:num_to_mask]]
        self.original_requires_grad = {}
        for name, param in model.named_parameters():
            if name in self.masked_layers:
                self.original_requires_grad[name] = True
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    def on_step_end(self, args, state, control, model, **kwargs):
        for name, param in model.named_parameters():
            if name in self.masked_layers:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
