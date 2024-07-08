# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

class ElasticBiSpiking(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        elif num_bits == 1 or num_bits == 2:
            Qn = -1
            Qp = 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()  ################################## binary
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)  ###################### ternary
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class AlphaInit(nn.Parameter):
    def __init__(self, tensor,  requires_grad=True):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor, requires_grad=requires_grad)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


class SpikeLinear(nn.Linear):

    def __init__(self,  *kargs, symmetric=True, bias=True, config=None):
        super(SpikeLinear, self).__init__(*kargs,bias=True)
        self.weight_bits = config.weight_bits
        self.quantize_act = config.quantize_act
        
        self.register_buffer('weight_clip_val', torch.tensor([config.clip_val]))
        
        self.input_bits = config.input_bits
        
        self.T = config.T
        self.act_clip_val = nn.ParameterList([AlphaInit(torch.tensor(1.0), requires_grad =False) for i in range(self.T)])
        self.act_quantizer = ElasticBiSpiking


    def forward(self, input):
        # quantize weight
        assert len(self.weight.size()) == 2

        weight = self.weight
        mem = torch.zeros_like(input[0]).cuda()
        output = torch.zeros_like(input).cuda()
        mem_old = 0
        for i in range(self.T):
            if i == 0:
                mem = input[0]
            else:
                mem = mem_old * 0.25 * (self.act_clip_val[i-1].detach() - output[i-1].detach()) + input[i]

            output[i] = self.act_quantizer.apply(mem, self.act_clip_val[i], self.input_bits, True)
            mem_old = mem.clone()
        
        out = nn.functional.linear(output, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):

    def __init__(self,  *kargs,padding_idx=None, config = None):
        print('init quantize emb')
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            quan_weights_no_grad = scaling_factor * (torch.sign(real_weights/scaling_factor))
        elif self.weight_bits == 2:
            scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
        else:
            raise NotImplementedError

        weight = quan_weights_no_grad.detach() - real_weights.detach() + real_weights

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out





