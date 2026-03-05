#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键修复训练错误 - 应用推荐设置
"""

import json
import sys

print("=" * 60)
print("一键修复训练错误")
print("=" * 60)

# 读取 notebook
with open('machine_translation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

# 找到超参数配置单元格
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 修改超参数
        if 'BATCH_SIZE' in source and 'EPOCHS_SEQ2SEQ' in source:
            print(f"\n找到超参数配置单元格 (索引 {i})")
            
            original = source
            
            # 减小批次大小
            if 'BATCH_SIZE = 64' in source:
                source = source.replace('BATCH_SIZE = 64', 'BATCH_SIZE = 16')
                changes.append("BATCH_SIZE: 64 → 16")
            elif 'BATCH_SIZE = 32' in source:
                source = source.replace('BATCH_SIZE = 32', 'BATCH_SIZE = 16')
                changes.append("BATCH_SIZE: 32 → 16")
            
            # 减少训练轮数
            if 'EPOCHS_SEQ2SEQ = 20' in source:
                source = source.replace('EPOCHS_SEQ2SEQ = 20', 'EPOCHS_SEQ2SEQ = 10')
                changes.append("EPOCHS_SEQ2SEQ: 20 → 10")
            
            if 'EPOCHS_TRANSFORMER = 15' in source:
                source = source.replace('EPOCHS_TRANSFORMER = 15', 'EPOCHS_TRANSFORMER = 10')
                changes.append("EPOCHS_TRANSFORMER: 15 → 10")
            
            # 降低学习率
            if 'LEARNING_RATE_SEQ2SEQ = 0.001' in source:
                source = source.replace('LEARNING_RATE_SEQ2SEQ = 0.001', 'LEARNING_RATE_SEQ2SEQ = 0.0005')
                changes.append("LEARNING_RATE_SEQ2SEQ: 0.001 → 0.0005")
            
            if source != original:
                cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
                print("已应用修改")

if changes:
    print("\n应用的修改：")
    for change in changes:
        print(f"  ✓ {change}")
    
    # 保存
    with open('machine_translation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print("\n" + "=" * 60)
    print("修复完成！")
    print("=" * 60)
    print("\n下一步操作：")
    print("1. 在 Jupyter 中点击 'Kernel' → 'Restart Kernel'")
    print("2. 从第一个单元格开始运行")
    print("3. 确保运行 GPU 诊断单元格")
    print("\n这些设置可以大大提高训练稳定性！")
else:
    print("\n未找到需要修改的参数，或参数已经是推荐值")

