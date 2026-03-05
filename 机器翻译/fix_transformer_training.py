import json

with open('machine_translation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 找到 Transformer 训练单元格
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 修改 train_transformer_epoch 函数
        if 'def train_transformer_epoch' in source and 'loss.backward()' in source:
            print(f"找到 Transformer 训练函数，索引: {i}")
            
            # 添加 NaN 检测
            new_source = source.replace(
                '        loss = criterion(output, trg_target)\n        loss.backward()',
                '''        loss = criterion(output, trg_target)
        
        # 检查损失是否为 NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: 检测到 NaN/Inf 损失值，跳过此批次")
            optimizer.zero_grad()
            continue
        
        loss.backward()'''
            )
            
            # 更新单元格
            cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
            print("已添加 NaN 检测")
            
        # 在训练循环前添加 GPU 配置
        if 'for epoch in range(EPOCHS_TRANSFORMER)' in source and 'torch.cuda.empty_cache' not in source:
            print(f"找到 Transformer 训练循环，索引: {i}")
            
            new_source = source.replace(
                'for epoch in range(EPOCHS_TRANSFORMER):',
                '''# 清理 GPU 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

for epoch in range(EPOCHS_TRANSFORMER):'''
            )
            
            cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
            print("已添加 GPU 缓存清理")

# 保存
with open('machine_translation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n修复完成！")
print("建议操作：")
print("1. 重启 Jupyter Kernel")
print("2. 重新运行所有单元格")
print("3. 如果仍有问题，减小 BATCH_SIZE")
