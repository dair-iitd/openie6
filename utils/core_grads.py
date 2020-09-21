"""
Module to describe gradients
"""
from typing import Dict

from torch import nn

import ipdb

class GradInformation(nn.Module):

    def grad_norm(self, norm_type: float, opt_idx: float) -> Dict[str, int]:
        results = {}
        total_norm = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)

                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    # results['grad_{}_norm_{}'.format(norm_type, name)] = grad
                except Exception:
                    # this param had no grad
                    pass

        total_norm = total_norm ** (1. / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['opt_{}_grad_{}_norm_total'.format(opt_idx, norm_type)] = grad
        return results
