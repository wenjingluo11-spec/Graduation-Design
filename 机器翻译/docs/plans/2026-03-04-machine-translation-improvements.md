# 机器翻译模块改进实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 为毕业设计的机器翻译模块添加注意力可视化、错误分析、统计测试和交互式演示功能

**架构:** 在现有 Jupyter Notebook 基础上添加独立的分析和可视化单元格,保持模块化设计。每个新功能作为独立函数实现,可复用于 Seq2Seq 和 Transformer 模型。

**技术栈:** PyTorch, matplotlib, seaborn, numpy, scipy (统计测试)

---

## Task 1: 注意力可视化 - Seq2Seq Bahdanau Attention

**文件:**
- 修改: `machine_translation.ipynb` (在评估部分后添加新单元格)

**步骤 1: 修改 Seq2Seq 解码器以返回注意力权重**

在现有的 `Seq2SeqDecoder` 类中,修改 `forward` 方法返回注意力权重:

```python
def forward(self, input, hidden, encoder_outputs):
    # ... 现有代码 ...
    attn_weights = self.attention(hidden, encoder_outputs)
    # ... 现有代码 ...
    return output, hidden, attn_weights  # 添加 attn_weights 返回值
```

**步骤 2: 创建注意力可视化函数**

```python
def visualize_attention(input_sentence, output_words, attention_weights, save_path=None):
    """
    可视化注意力权重热力图

    Args:
        input_sentence: 源语言句子 (list of words)
        output_words: 目标语言单词 (list of words)
        attention_weights: 注意力权重矩阵 (target_len, source_len)
        save_path: 保存路径 (可选)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights,
                xticklabels=input_sentence,
                yticklabels=output_words,
                cmap='YlOrRd',
                cbar_kws={'label': '注意力权重'})
    plt.xlabel('源语言 (英语)')
    plt.ylabel('目标语言 (西班牙语)')
    plt.title('Seq2Seq 注意力权重可视化')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**步骤 3: 修改翻译函数以收集注意力权重**

```python
def translate_with_attention(model, sentence, src_vocab, tgt_vocab, device, max_length=50):
    """
    翻译句子并返回注意力权重

    Returns:
        translated_words: 翻译结果
        attention_matrix: 注意力权重矩阵 (numpy array)
    """
    model.eval()
    with torch.no_grad():
        # 编码
        tokens = [src_vocab.get(word, src_vocab['<UNK>']) for word in sentence.split()]
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
        encoder_outputs, hidden = model.encoder(input_tensor)

        # 解码并收集注意力
        decoder_input = torch.tensor([[tgt_vocab['<SOS>']]], device=device)
        translated_words = []
        attention_weights = []

        for _ in range(max_length):
            output, hidden, attn = model.decoder(decoder_input, hidden, encoder_outputs)
            attention_weights.append(attn.squeeze().cpu().numpy())

            topv, topi = output.topk(1)
            if topi.item() == tgt_vocab['<EOS>']:
                break

            translated_words.append(topi.item())
            decoder_input = topi.detach()

        return translated_words, np.array(attention_weights)
```

**步骤 4: 创建示例可视化单元格**

```python
# 选择测试集中的示例句子
example_idx = 42  # 可调整
example_sentence = test_pairs[example_idx][0]

# 翻译并获取注意力权重
translated_ids, attention_matrix = translate_with_attention(
    seq2seq_model, example_sentence, src_vocab, tgt_vocab, DEVICE
)

# 转换为单词
input_words = example_sentence.split()
output_words = [idx_to_tgt[idx] for idx in translated_ids]

# 可视化
visualize_attention(
    input_words,
    output_words,
    attention_matrix,
    save_path='attention_seq2seq_example.png'
)

print(f"源句子: {example_sentence}")
print(f"翻译结果: {' '.join(output_words)}")
print(f"参考翻译: {test_pairs[example_idx][1]}")
```

**步骤 5: 验证输出**

运行单元格,确认:
- 热力图正确显示
- x 轴为源语言单词
- y 轴为目标语言单词
- 对角线附近权重较高(单调对齐)
- 图片保存成功

---

## Task 2: 注意力可视化 - Transformer Multi-Head Attention

**文件:**
- 修改: `machine_translation.ipynb` (在 Task 1 后添加新单元格)

**步骤 1: 修改 Transformer 模型以返回注意力权重**

```python
# 在 TransformerModel 类中添加方法
def forward_with_attention(self, src, tgt, src_mask=None, tgt_mask=None):
    """
    前向传播并返回注意力权重

    Returns:
        output: 模型输出
        encoder_attn: 编码器自注意力权重 (list of tensors)
        decoder_self_attn: 解码器自注意力权重 (list of tensors)
        decoder_cross_attn: 解码器交叉注意力权重 (list of tensors)
    """
    # 需要修改 nn.Transformer 以暴露注意力权重
    # 或使用自定义 Transformer 实现
    pass  # 实现细节取决于现有代码结构
```

**步骤 2: 创建多头注意力可视化函数**

```python
def visualize_multihead_attention(input_sentence, output_words, attention_weights,
                                   num_heads=8, layer_idx=0, save_path=None):
    """
    可视化 Transformer 多头注意力

    Args:
        attention_weights: (num_heads, target_len, source_len)
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for head in range(num_heads):
        sns.heatmap(attention_weights[head],
                    xticklabels=input_sentence,
                    yticklabels=output_words,
                    cmap='viridis',
                    ax=axes[head],
                    cbar=True)
        axes[head].set_title(f'Head {head + 1}')
        axes[head].set_xlabel('源语言')
        axes[head].set_ylabel('目标语言')

    plt.suptitle(f'Transformer Layer {layer_idx} 多头注意力可视化', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**步骤 3: 创建平均注意力可视化**

```python
def visualize_average_attention(input_sentence, output_words, attention_weights, save_path=None):
    """
    可视化所有头的平均注意力权重

    Args:
        attention_weights: (num_heads, target_len, source_len)
    """
    avg_attention = attention_weights.mean(axis=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_attention,
                xticklabels=input_sentence,
                yticklabels=output_words,
                cmap='Blues',
                cbar_kws={'label': '平均注意力权重'})
    plt.xlabel('源语言 (英语)')
    plt.ylabel('目标语言 (西班牙语)')
    plt.title('Transformer 平均注意力权重')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**步骤 4: 验证输出**

运行示例,确认:
- 8 个子图正确显示
- 不同头关注不同模式
- 平均注意力图合理

---

## Task 3: 错误分析模块

**文件:**
- 修改: `machine_translation.ipynb` (添加新的分析单元格)

**步骤 1: 创建按句子长度分析函数**

```python
def analyze_bleu_by_length(model, test_pairs, src_vocab, tgt_vocab, device,
                           translate_fn, length_bins=[5, 10, 15, 20, 30]):
    """
    按源句子长度分析 BLEU 分数

    Args:
        length_bins: 长度区间边界

    Returns:
        results: dict with keys 'length_range', 'bleu_scores', 'count'
    """
    from collections import defaultdict

    length_groups = defaultdict(list)

    for src, tgt in test_pairs:
        src_len = len(src.split())
        # 分配到长度区间
        for i in range(len(length_bins)):
            if i == 0 and src_len < length_bins[0]:
                length_groups[f'<{length_bins[0]}'].append((src, tgt))
                break
            elif i < len(length_bins) - 1 and length_bins[i] <= src_len < length_bins[i+1]:
                length_groups[f'{length_bins[i]}-{length_bins[i+1]}'].append((src, tgt))
                break
            elif i == len(length_bins) - 1 and src_len >= length_bins[-1]:
                length_groups[f'>={length_bins[-1]}'].append((src, tgt))
                break

    results = {'length_range': [], 'bleu_4': [], 'count': []}

    for length_range in sorted(length_groups.keys()):
        pairs = length_groups[length_range]
        bleu_score = evaluate_bleu(model, pairs, src_vocab, tgt_vocab, device, translate_fn)
        results['length_range'].append(length_range)
        results['bleu_4'].append(bleu_score['bleu_4'])
        results['count'].append(len(pairs))

    return results
```

**步骤 2: 创建可视化函数**

```python
def plot_bleu_by_length(results, model_name, save_path=None):
    """
    绘制 BLEU 分数随句子长度变化的图表
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(results['length_range']))
    ax1.bar(x, results['bleu_4'], alpha=0.7, color='steelblue', label='BLEU-4')
    ax1.set_xlabel('句子长度区间')
    ax1.set_ylabel('BLEU-4 分数', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results['length_range'], rotation=45)
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, results['count'], color='coral', marker='o', linewidth=2, label='样本数量')
    ax2.set_ylabel('样本数量', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    plt.title(f'{model_name} - BLEU 分数随句子长度变化')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

**步骤 3: 创建最佳/最差案例分析函数**

```python
def find_best_worst_cases(model, test_pairs, src_vocab, tgt_vocab, device,
                          translate_fn, n=5):
    """
    找出翻译最好和最差的案例

    Returns:
        best_cases: list of (src, ref, hyp, bleu_4)
        worst_cases: list of (src, ref, hyp, bleu_4)
    """
    cases = []

    for src, tgt in test_pairs:
        # 翻译单个句子
        hyp = translate_fn(model, src, src_vocab, tgt_vocab, device)

        # 计算单句 BLEU-4
        reference = [tgt.split()]
        hypothesis = hyp.split()
        bleu_4 = compute_bleu_scores([reference], [hypothesis])['bleu_4']

        cases.append({
            'source': src,
            'reference': tgt,
            'hypothesis': hyp,
            'bleu_4': bleu_4
        })

    # 排序
    cases_sorted = sorted(cases, key=lambda x: x['bleu_4'], reverse=True)

    return cases_sorted[:n], cases_sorted[-n:]
```

**步骤 4: 创建案例展示函数**

```python
def display_translation_cases(cases, title):
    """
    格式化展示翻译案例
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")

    for i, case in enumerate(cases, 1):
        print(f"案例 {i} (BLEU-4: {case['bleu_4']:.4f})")
        print(f"  源句子: {case['source']}")
        print(f"  参考翻译: {case['reference']}")
        print(f"  模型翻译: {case['hypothesis']}")
        print()
```

**步骤 5: 运行完整分析**

```python
# 按长度分析
print("正在分析 Seq2Seq 模型...")
seq2seq_length_results = analyze_bleu_by_length(
    seq2seq_model, test_pairs, src_vocab, tgt_vocab, DEVICE, translate_sentence
)
plot_bleu_by_length(seq2seq_length_results, 'Seq2Seq', 'bleu_by_length_seq2seq.png')

print("正在分析 Transformer 模型...")
tf_length_results = analyze_bleu_by_length(
    transformer_model, test_pairs, src_vocab, tgt_vocab, DEVICE, translate_sentence_tf
)
plot_bleu_by_length(tf_length_results, 'Transformer', 'bleu_by_length_transformer.png')

# 最佳/最差案例
print("正在查找最佳/最差翻译案例...")
best_seq2seq, worst_seq2seq = find_best_worst_cases(
    seq2seq_model, test_pairs[:100], src_vocab, tgt_vocab, DEVICE, translate_sentence, n=5
)

display_translation_cases(best_seq2seq, "Seq2Seq 最佳翻译案例 (Top 5)")
display_translation_cases(worst_seq2seq, "Seq2Seq 最差翻译案例 (Bottom 5)")
```

**步骤 6: 验证输出**

确认:
- 长度分析图表正确显示双 y 轴
- 最佳案例 BLEU 分数接近 1.0
- 最差案例 BLEU 分数接近 0.0
- 案例展示格式清晰

---

## Task 4: 统计显著性测试

**文件:**
- 修改: `machine_translation.ipynb` (添加统计测试单元格)

**步骤 1: 创建配对 t 检验函数**

```python
from scipy import stats

def paired_t_test(model1_scores, model2_scores, model1_name, model2_name):
    """
    对两个模型的 BLEU 分数进行配对 t 检验

    Args:
        model1_scores: list of BLEU scores for model 1
        model2_scores: list of BLEU scores for model 2

    Returns:
        dict with t_statistic, p_value, significant, mean_diff, ci_lower, ci_upper
    """
    import numpy as np

    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)

    # 计算差值的置信区间
    differences = np.array(model1_scores) - np.array(model2_scores)
    mean_diff = differences.mean()
    std_diff = differences.std(ddof=1)
    n = len(differences)

    # 95% 置信区间
    ci = stats.t.interval(0.95, n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': mean_diff,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'model1_name': model1_name,
        'model2_name': model2_name
    }
```

**步骤 2: 计算每个测试样本的 BLEU 分数**

```python
def compute_per_sample_bleu(model, test_pairs, src_vocab, tgt_vocab, device, translate_fn):
    """
    计算测试集中每个样本的 BLEU-4 分数

    Returns:
        list of BLEU-4 scores
    """
    bleu_scores = []

    for src, tgt in test_pairs:
        hyp = translate_fn(model, src, src_vocab, tgt_vocab, device)
        reference = [tgt.split()]
        hypothesis = hyp.split()
        bleu_4 = compute_bleu_scores([reference], [hypothesis])['bleu_4']
        bleu_scores.append(bleu_4)

    return bleu_scores
```

**步骤 3: 运行统计测试**

```python
print("正在计算每个样本的 BLEU 分数...")
seq2seq_scores = compute_per_sample_bleu(
    seq2seq_model, test_pairs, src_vocab, tgt_vocab, DEVICE, translate_sentence
)

transformer_scores = compute_per_sample_bleu(
    transformer_model, test_pairs, src_vocab, tgt_vocab, DEVICE, translate_sentence_tf
)

print("正在进行配对 t 检验...")
test_result = paired_t_test(
    transformer_scores, seq2seq_scores,
    'Transformer', 'Seq2Seq'
)
```

**步骤 4: 创建结果展示函数**

```python
def display_statistical_test(result):
    """
    格式化展示统计测试结果
    """
    print(f"\n{'='*80}")
    print(f"配对 t 检验: {result['model1_name']} vs {result['model2_name']}")
    print(f"{'='*80}\n")

    print(f"t 统计量: {result['t_statistic']:.4f}")
    print(f"p 值: {result['p_value']:.6f}")
    print(f"显著性 (α=0.05): {'是' if result['significant'] else '否'}")
    print(f"\n平均差值: {result['mean_diff']:.4f}")
    print(f"95% 置信区间: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    if result['significant']:
        if result['mean_diff'] > 0:
            print(f"\n结论: {result['model1_name']} 显著优于 {result['model2_name']}")
        else:
            print(f"\n结论: {result['model2_name']} 显著优于 {result['model1_name']}")
    else:
        print(f"\n结论: 两个模型之间没有显著差异")

display_statistical_test(test_result)
```

**步骤 5: 创建 BLEU 分数分布可视化**

```python
def plot_bleu_distribution(scores1, scores2, name1, name2, save_path=None):
    """
    绘制两个模型的 BLEU 分数分布对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(scores1, bins=30, alpha=0.6, label=name1, color='steelblue')
    axes[0].hist(scores2, bins=30, alpha=0.6, label=name2, color='coral')
    axes[0].set_xlabel('BLEU-4 分数')
    axes[0].set_ylabel('频数')
    axes[0].set_title('BLEU 分数分布')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 箱线图
    axes[1].boxplot([scores1, scores2], labels=[name1, name2])
    axes[1].set_ylabel('BLEU-4 分数')
    axes[1].set_title('BLEU 分数箱线图')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_bleu_distribution(
    transformer_scores, seq2seq_scores,
    'Transformer', 'Seq2Seq',
    'bleu_distribution_comparison.png'
)
```

**步骤 6: 验证输出**

确认:
- t 检验结果合理
- p 值正确计算
- 置信区间不包含 0 时显著
- 分布图清晰展示差异

---

## Task 5: 交互式翻译演示

**文件:**
- 修改: `machine_translation.ipynb` (添加交互式单元格)

**步骤 1: 创建交互式翻译函数**

```python
def interactive_translation_demo(seq2seq_model, transformer_model,
                                 src_vocab, tgt_vocab, device):
    """
    交互式翻译演示
    """
    print("="*80)
    print("机器翻译交互式演示 (英语 → 西班牙语)")
    print("="*80)
    print("输入 'quit' 退出\n")

    while True:
        # 获取用户输入
        input_sentence = input("请输入英语句子: ").strip()

        if input_sentence.lower() == 'quit':
            print("退出演示")
            break

        if not input_sentence:
            continue

        # 预处理
        input_sentence = input_sentence.lower()

        print(f"\n{'─'*80}")
        print(f"源句子: {input_sentence}")
        print(f"{'─'*80}\n")

        # Seq2Seq 翻译
        print("【Seq2Seq + Attention】")
        seq2seq_greedy = translate_sentence(seq2seq_model, input_sentence,
                                            src_vocab, tgt_vocab, device)
        seq2seq_beam = translate_sentence_beam(seq2seq_model, input_sentence,
                                               src_vocab, tgt_vocab, device)
        print(f"  贪心解码: {seq2seq_greedy}")
        print(f"  Beam Search: {seq2seq_beam}")

        # Transformer 翻译
        print("\n【Transformer】")
        tf_greedy = translate_sentence_tf(transformer_model, input_sentence,
                                          src_vocab, tgt_vocab, device)
        tf_beam = translate_sentence_tf_beam(transformer_model, input_sentence,
                                             src_vocab, tgt_vocab, device)
        print(f"  贪心解码: {tf_greedy}")
        print(f"  Beam Search: {tf_beam}")
        print()
```

**步骤 2: 创建批量测试函数**

```python
def batch_translation_demo(sentences, seq2seq_model, transformer_model,
                          src_vocab, tgt_vocab, device):
    """
    批量翻译演示

    Args:
        sentences: list of (english, spanish) tuples
    """
    results = []

    for eng, spa in sentences:
        seq2seq_trans = translate_sentence_beam(seq2seq_model, eng,
                                                src_vocab, tgt_vocab, device)
        tf_trans = translate_sentence_tf_beam(transformer_model, eng,
                                              src_vocab, tgt_vocab, device)

        results.append({
            'source': eng,
            'reference': spa,
            'seq2seq': seq2seq_trans,
            'transformer': tf_trans
        })

    return results
```

**步骤 3: 创建结果展示函数**

```python
def display_batch_results(results):
    """
    展示批量翻译结果
    """
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"示例 {i}")
        print(f"{'='*80}")
        print(f"源句子:     {result['source']}")
        print(f"参考翻译:   {result['reference']}")
        print(f"Seq2Seq:    {result['seq2seq']}")
        print(f"Transformer: {result['transformer']}")
```

**步骤 4: 准备演示句子**

```python
# 准备一些有趣的测试句子
demo_sentences = [
    ("i love you", "te quiero"),
    ("how are you", "como estas"),
    ("good morning", "buenos dias"),
    ("thank you very much", "muchas gracias"),
    ("where is the bathroom", "donde esta el bano"),
]

print("批量翻译演示:")
demo_results = batch_translation_demo(
    demo_sentences, seq2seq_model, transformer_model,
    src_vocab, tgt_vocab, DEVICE
)
display_batch_results(demo_results)
```

**步骤 5: 启动交互式演示**

```python
# 启动交互式演示
interactive_translation_demo(
    seq2seq_model, transformer_model,
    src_vocab, tgt_vocab, DEVICE
)
```

**步骤 6: 验证功能**

测试:
- 输入常见句子,检查翻译质量
- 验证贪心和 Beam Search 结果不同
- 确认两个模型都能正常工作
- 测试 'quit' 命令退出

---

## 执行选项

计划已完成并保存。两种执行方式:

**1. 子代理驱动 (当前会话)** - 我为每个任务派发新的子代理,任务间进行代码审查,快速迭代

**2. 并行会话 (独立会话)** - 在新会话中使用 executing-plans 技能,批量执行并设置检查点

你希望使用哪种方式?
