#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 DiffKG 模块导入和核心功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("测试 DiffKG 模块（方案一：纯 PyTorch 实现）")
print("=" * 60)

# 测试 1: 导入模块
print("\n[测试 1] 导入 DiffKG 模块...")
try:
    import warnings
    warnings.simplefilter('always')
    from models.general.DiffKG import DiffKG, scatter_sum, scatter_softmax, USE_TORCH_SCATTER
    print(f"✓ DiffKG 模块导入成功")
    print(f"  使用 torch_scatter: {USE_TORCH_SCATTER}")
    if not USE_TORCH_SCATTER:
        print("  → 正在使用纯 PyTorch 实现（这是正常的）")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试 2: 测试 scatter 函数
print("\n[测试 2] 测试 scatter_sum 和 scatter_softmax 函数...")
try:
    import torch
    
    # 测试 scatter_sum
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx = torch.tensor([0, 0, 1, 1])
    result_sum = scatter_sum(src, idx)
    expected_sum = torch.tensor([3.0, 7.0])  # [1+2, 3+4]
    assert torch.allclose(result_sum, expected_sum), f"scatter_sum 结果不正确: {result_sum} vs {expected_sum}"
    print(f"✓ scatter_sum 测试通过: {result_sum.tolist()}")
    
    # 测试 scatter_softmax
    result_softmax = scatter_softmax(src, idx)
    # 验证 softmax 性质：每个组内和为1
    group0_sum = result_softmax[0] + result_softmax[1]
    group1_sum = result_softmax[2] + result_softmax[3]
    assert abs(group0_sum.item() - 1.0) < 1e-5, f"组0 softmax和不为1: {group0_sum}"
    assert abs(group1_sum.item() - 1.0) < 1e-5, f"组1 softmax和不为1: {group1_sum}"
    print(f"✓ scatter_softmax 测试通过: {result_softmax.tolist()}")
    
except Exception as e:
    print(f"✗ scatter 函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: 测试 RGAT 组件
print("\n[测试 3] 测试 RGAT 组件...")
try:
    from models.general.DiffKG import RGAT
    import torch.nn as nn
    
    latdim = 32
    n_hops = 1
    rgat = RGAT(latdim, n_hops, mess_dropout_rate=0.1)
    
    # 创建测试数据
    n_entities = 10
    n_relations = 3
    entity_emb = torch.randn(n_entities, latdim)
    relation_emb = torch.randn(n_relations, latdim)
    
    # 创建简单的 KG 边
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # 3条边
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    kg = (edge_index, edge_type)
    
    # 前向传播
    output = rgat.forward(entity_emb, relation_emb, kg, mess_dropout=False)
    
    assert output.shape == entity_emb.shape, f"输出形状不正确: {output.shape} vs {entity_emb.shape}"
    print(f"✓ RGAT 前向传播测试通过: 输入形状 {entity_emb.shape}, 输出形状 {output.shape}")
    
except Exception as e:
    print(f"✗ RGAT 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有核心功能测试通过！")
print("=" * 60)
print("\n说明：")
print("- 纯 PyTorch 实现已正常工作")
print("- 即使没有安装 torch_scatter，代码也可以运行")
print("- 如果后续需要提升性能，可以尝试安装 torch_scatter")
print("\n现在可以尝试运行完整的训练：")
print("  python main.py --model_name DiffKG --dataset Grocery_and_Gourmet_Food ...")


