#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 ReChorus 和 DiffKG 所需的所有依赖
"""

import sys

required_modules = [
    'torch',
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'tqdm',
    'yaml',
]

missing_modules = []

print("检查依赖...")
for module in required_modules:
    try:
        if module == 'sklearn':
            __import__('sklearn')
        elif module == 'yaml':
            __import__('yaml')
        else:
            __import__(module)
        print(f"✓ {module}")
    except ImportError:
        print(f"✗ {module} - 缺失")
        missing_modules.append(module)

if missing_modules:
    print(f"\n缺少以下模块: {', '.join(missing_modules)}")
    print("请运行: pip install " + " ".join(missing_modules))
    sys.exit(1)
else:
    print("\n✓ 所有依赖都已安装！")
    
# 检查 torch_scatter（可选）
try:
    from torch_scatter import scatter_sum, scatter_softmax
    print("✓ torch_scatter - 已安装（将使用高效实现）")
except ImportError:
    print("⚠ torch_scatter - 未安装（将使用纯 PyTorch 实现，性能稍慢但可用）")

print("\n现在可以运行 DiffKG 训练了！")


