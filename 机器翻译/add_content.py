#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新增内容整合到 machine_translation.ipynb
"""

import json
import sys

def create_markdown_cell(content):
    """创建 Markdown 单元格"""
    lines = content.split('\n')
    # 确保每行末尾有换行符（除了最后一行）
    formatted_lines = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        formatted_lines.append(lines[-1])
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": formatted_lines
    }

def create_code_cell(content):
    """创建 Code 单元格"""
    lines = content.split('\n')
    formatted_lines = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        formatted_lines.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": formatted_lines
    }

print("正在读取原始 notebook...")
with open('machine_translation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 找到插入位置（Section 8 之前）
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '第8部分' in source_text or '## 第8部分' in source_text:
            insert_idx = i
            break

if insert_idx is None:
    print("错误：未找到 Section 8")
    sys.exit(1)

print(f"找到插入位置：索引 {insert_idx}")
print(f"原始 notebook 有 {len(nb['cells'])} 个单元格")

# 准备新增单元格
new_cells = []

print("创建新单元格...")

# ===== Section 7.4: Seq2Seq 注意力权重可视化 =====
new_cells.append(create_markdown_cell(
    "## 7.4 Seq2Seq 注意力权重可视化\n\n"
    "可视化 Seq2Seq 模型的 Bahdanau 注意力机制，展示模型在翻译时如何关注源语言的不同部分。"
))

# 函数定义 - 第1部分
new_cells.append(create_code_cell("""def translate_with_attention(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    \"\"\"翻译句子并返回注意力权重\"\"\"
    model.eval()
    with torch.no_grad():
        try:
            if not src_sentence or not src_sentence.strip():
                print("警告: 源句子为空")
                return [], np.zeros((0, 0))
            
            src_tokens = src_sentence.lower().split()
            src_indices = [src_vocab.word2idx.get(word, src_vocab.word2idx['<unk>']) for word in src_tokens]
            src_indices = [src_vocab.word2idx['<sos>']] + src_indices + [src_vocab.word2idx['<eos>']]
            src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)
            src_len = len(src_indices)
            
            encoder_outputs, hidden = model.encoder(src_tensor)
            decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]], device=device)
            translated_ids = []
            attention_weights = []
            
            for _ in range(max_len):
                output, hidden, attn = model.decoder(decoder_input, hidden, encoder_outputs)
                attention_weights.append(attn.squeeze(0).cpu().numpy())
                topv, topi = output.topk(1)
                predicted_id = topi.item()
                if predicted_id == tgt_vocab.word2idx['<eos>']:
                    break
                translated_ids.append(predicted_id)
                decoder_input = topi.detach()
            
            if not attention_weights:
                return translated_ids, np.zeros((0, src_len))
            attention_matrix = np.array(attention_weights)
            return translated_ids, attention_matrix
        except Exception as e:
            print(f"翻译失败: {type(e).__name__}: {e}")
            return [], np.zeros((0, 0))


def visualize_attention(input_words, output_words, attention_weights, save_path=None,
                       width_per_word=0.6, height_per_word=0.5):
    \"\"\"可视化注意力权重热力图\"\"\"
    if attention_weights.size == 0 or len(input_words) == 0 or len(output_words) == 0:
        print("警告: 注意力权重或词列表为空，跳过可视化")
        return
    if attention_weights.shape != (len(output_words), len(input_words)):
        print(f"警告: 注意力矩阵形状 {attention_weights.shape} 与词列表长度不匹配")
        return
    
    width = max(8, len(input_words) * width_per_word)
    height = max(6, len(output_words) * height_per_word)
    plt.figure(figsize=(width, height))
    sns.heatmap(attention_weights, xticklabels=input_words, yticklabels=output_words,
                cmap='YlOrRd', cbar_kws={'label': '注意力权重'})
    plt.xlabel('源语言 (英语)')
    plt.ylabel('目标语言 (西班牙语)')
    plt.title('Seq2Seq Bahdanau 注意力权重可视化')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print("Seq2Seq 注意力可视化函数已定义")"""))

# 示例可视化
new_cells.append(create_code_cell("""# 选择测试集中的示例句子
example_idx = 42
example_pair = test_pairs[example_idx]
src_sentence = example_pair[0]
ref_translation = example_pair[1]

print("=" * 60)
print(f"示例 {example_idx}")
print("=" * 60)
print(f"源句子: {src_sentence}")
print(f"参考翻译: {ref_translation}")

translated_ids, attention_matrix = translate_with_attention(
    seq2seq_model, src_sentence, eng_vocab, spa_vocab, DEVICE
)

src_words = src_sentence.split()
tgt_words = [spa_vocab.idx2word[idx] for idx in translated_ids
             if idx not in [spa_vocab.word2idx['<sos>'], spa_vocab.word2idx['<eos>'], spa_vocab.word2idx['<pad>']]]

print(f"模型翻译: {' '.join(tgt_words)}")

if len(tgt_words) > 0 and attention_matrix.size > 0:
    visualize_attention(src_words, tgt_words, attention_matrix, save_path='attention_seq2seq_example.png')
    print("\n注意力热力图已保存为 'attention_seq2seq_example.png'")"""))

print(f"已创建 {len(new_cells)} 个单元格")

# 插入新单元格
nb['cells'] = nb['cells'][:insert_idx] + new_cells + nb['cells'][insert_idx:]

print(f"新 notebook 有 {len(nb['cells'])} 个单元格")

# 保存备份
import shutil
shutil.copy('machine_translation.ipynb', 'machine_translation_backup.ipynb')
print("已创建备份: machine_translation_backup.ipynb")

# 保存修改后的 notebook
with open('machine_translation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ 成功添加 Section 7.4 内容到 notebook")
print(f"✓ 新增了 {len(new_cells)} 个单元格")
