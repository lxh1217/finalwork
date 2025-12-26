#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
辅助脚本：自动检测环境并安装匹配的 torch_scatter
"""

import subprocess
import sys
import torch

def get_pytorch_info():
    """获取 PyTorch 版本和 CUDA 信息"""
    version = torch.__version__
    has_cuda = torch.cuda.is_available()
    cuda_version = None
    
    if has_cuda:
        try:
            cuda_version = torch.version.cuda
            # 提取主版本号，如 11.8 -> cu118, 12.1 -> cu121
            major, minor = cuda_version.split('.')[:2]
            cuda_tag = f"cu{major}{minor}"
        except:
            cuda_tag = "cu118"  # 默认
    else:
        cuda_tag = "cpu"
    
    return version, has_cuda, cuda_tag, cuda_version

def install_torch_scatter():
    """尝试安装 torch_scatter"""
    print("=" * 60)
    print("torch_scatter 安装助手")
    print("=" * 60)
    
    version, has_cuda, cuda_tag, cuda_ver = get_pytorch_info()
    
    print(f"\n检测到的环境信息：")
    print(f"  PyTorch 版本: {version}")
    print(f"  CUDA 可用: {has_cuda}")
    if has_cuda:
        print(f"  CUDA 版本: {cuda_ver}")
    print(f"  标签: {cuda_tag}")
    
    # 提取 PyTorch 主版本号（如 2.0.0 -> 2.0.0）
    pytorch_version = version.split('+')[0]  # 移除 +cu118 等后缀
    
    print(f"\n尝试安装 torch_scatter...")
    
    # 方法1: 从 PyG 官方源安装
    urls = [
        f"https://data.pyg.org/whl/torch-{pytorch_version}+{cuda_tag}.html",
        f"https://data.pyg.org/whl/torch-{pytorch_version}+cpu.html",  # CPU 备选
    ]
    
    for url in urls:
        print(f"\n尝试从: {url}")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "torch-scatter", "-f", url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✓ 安装成功！")
                return True
            else:
                print(f"✗ 安装失败: {result.stderr[:200]}")
        except Exception as e:
            print(f"✗ 错误: {str(e)[:200]}")
    
    # 方法2: 尝试直接安装（可能从 PyPI 获取）
    print(f"\n尝试从 PyPI 直接安装...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "torch-scatter"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ 从 PyPI 安装成功！")
            return True
        else:
            print(f"✗ PyPI 安装失败")
    except Exception as e:
        print(f"✗ 错误: {str(e)[:200]}")
    
    print("\n" + "=" * 60)
    print("所有安装方法都失败了。")
    print("\n建议：")
    print("1. 代码已经包含纯 PyTorch 实现，可以直接运行（性能稍慢）")
    print("2. 或者手动安装 Visual C++ Build Tools 后从源码编译")
    print("3. 或者使用 conda 环境: conda install pytorch-scatter -c pyg")
    print("=" * 60)
    return False

if __name__ == "__main__":
    success = install_torch_scatter()
    sys.exit(0 if success else 1)


