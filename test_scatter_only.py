#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅测试 scatter 函数（不依赖其他 ReChorus 模块）
"""

import torch
import sys
import os

# 手动实现 scatter 函数（从 DiffKG.py 复制）
def scatter_sum(src, index, dim=0, dim_size=None):
    """Pure PyTorch implementation of scatter_sum."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    if src.dim() > 1:
        src_flat = src.view(src.size(0), -1)
        out_shape = (dim_size, src_flat.size(1))
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        index_expanded = index.view(-1, 1).expand(-1, src_flat.size(1))
        out = out.scatter_add_(0, index_expanded, src_flat)
        return out.view(dim_size, *src.shape[1:])
    else:
        out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(0, index, src)

def scatter_softmax(src, index, dim=0, dim_size=None):
    """Pure PyTorch implementation of scatter_softmax."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    
    max_val = torch.full((dim_size,), float('-inf'), dtype=src.dtype, device=src.device)
    if hasattr(torch.Tensor, 'scatter_reduce_'):
        try:
            max_val = max_val.scatter_reduce_(dim=0, index=index, src=src, reduce='amax', include_self=False)
        except:
            for i in range(dim_size):
                mask = (index == i)
                if mask.any():
                    max_val[i] = src[mask].max()
    else:
        for i in range(dim_size):
            mask = (index == i)
            if mask.any():
                max_val[i] = src[mask].max()
    
    max_val_per_element = max_val[index]
    exp_src = torch.exp(src - max_val_per_element)
    sum_exp = scatter_sum(exp_src, index, dim=dim, dim_size=dim_size)
    if sum_exp.dim() > 1:
        sum_exp_per_element = sum_exp[index]
    else:
        sum_exp_per_element = sum_exp[index]
    return exp_src / (sum_exp_per_element + 1e-8)

print("=" * 60)
print("测试纯 PyTorch scatter 实现（方案一）")
print("=" * 60)

# 测试 scatter_sum
print("\n[测试 1] scatter_sum...")
src = torch.tensor([1.0, 2.0, 3.0, 4.0])
idx = torch.tensor([0, 0, 1, 1])
result = scatter_sum(src, idx)
expected = torch.tensor([3.0, 7.0])  # [1+2, 3+4]
assert torch.allclose(result, expected), f"失败: {result} vs {expected}"
print(f"✓ 通过: {result.tolist()}")

# 测试 scatter_softmax
print("\n[测试 2] scatter_softmax...")
result_softmax = scatter_softmax(src, idx)
group0_sum = result_softmax[0] + result_softmax[1]
group1_sum = result_softmax[2] + result_softmax[3]
assert abs(group0_sum.item() - 1.0) < 1e-5, f"组0和不为1: {group0_sum}"
assert abs(group1_sum.item() - 1.0) < 1e-5, f"组1和不为1: {group1_sum}"
print(f"✓ 通过: {result_softmax.tolist()}")
print(f"  组0和: {group0_sum.item():.6f}, 组1和: {group1_sum.item():.6f}")

# 测试多维情况
print("\n[测试 3] 多维 scatter_sum...")
src_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
idx_2d = torch.tensor([0, 0, 1, 1])
result_2d = scatter_sum(src_2d, idx_2d)
expected_2d = torch.tensor([[4.0, 6.0], [12.0, 14.0]])  # [[1+3, 2+4], [5+7, 6+8]]
assert torch.allclose(result_2d, expected_2d), f"失败: {result_2d} vs {expected_2d}"
print(f"✓ 通过: {result_2d.tolist()}")

print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
print("\n结论：")
print("- 纯 PyTorch 实现工作正常")
print("- 即使没有 torch_scatter，核心功能也可以使用")
print("- 现在可以运行完整的 DiffKG 训练（需要先安装 ReChorus 的其他依赖）")


