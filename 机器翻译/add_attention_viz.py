import json

# Read the notebook
with open('machine_translation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Create new cells for attention visualization
new_cells = []

# Cell 1: Markdown header
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 第7.4部分：Seq2Seq 注意力权重可视化\n",
        "\n",
        "可视化 Bahdanau 注意力机制，展示模型在翻译每个目标词时关注源句子的哪些部分。"
    ]
}
new_cells.append(markdown_cell)

# Cell 2: Visualization functions
code_cell_1_source = """# ============================================================================
# 第7.4部分：Seq2Seq 注意力权重可视化
# ============================================================================

def translate_with_attention(model, src_tensor, max_len=MAX_SEQ_LEN):
    \"\"\"
    翻译句子并收集注意力权重

    参数:
        model: Seq2Seq模型
        src_tensor: 源语言张量 (src_len,)
        max_len: 最大解码长度

    返回:
        translated_ids: 翻译结果 (list of int)
        attention_matrix: 注意力权重矩阵 (target_len, source_len)
    \"\"\"
    model.eval()
    with torch.no_grad():
        src = src_tensor.unsqueeze(0).to(DEVICE)  # (1, src_len)
        encoder_outputs, hidden = model.encoder(src)

        decoder_input = torch.tensor([SOS_IDX], device=DEVICE)
        translated_ids = []
        attention_weights_list = []

        for _ in range(max_len):
            output, hidden, attn_weights = model.decoder(decoder_input, hidden, encoder_outputs)
            predicted_id = output.argmax(dim=1).item()

            if predicted_id == EOS_IDX:
                break

            translated_ids.append(predicted_id)
            # attn_weights: (batch, src_len) -> (src_len,)
            attention_weights_list.append(attn_weights.squeeze(0).cpu().numpy())
            decoder_input = torch.tensor([predicted_id], device=DEVICE)

        # 堆叠成矩阵: (target_len, source_len)
        attention_matrix = np.array(attention_weights_list) if attention_weights_list else np.array([[]])

    return translated_ids, attention_matrix


def visualize_attention(input_words, output_words, attention_weights, save_path=None):
    \"\"\"
    可视化注意力权重热力图

    参数:
        input_words: 源语言单词列表 (list of str)
        output_words: 目标语言单词列表 (list of str)
        attention_weights: 注意力权重矩阵 (target_len, source_len)
        save_path: 保存路径 (可选)
    \"\"\"
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制热力图
    sns.heatmap(
        attention_weights,
        xticklabels=input_words,
        yticklabels=output_words,
        cmap='YlOrRd',
        cbar_kws={'label': '注意力权重'},
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('源语言 (英语)', fontsize=13, fontweight='bold')
    ax.set_ylabel('目标语言 (西班牙语)', fontsize=13, fontweight='bold')
    ax.set_title('Seq2Seq Bahdanau 注意力权重可视化', fontsize=15, fontweight='bold', pad=20)

    # 旋转x轴标签以便阅读
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"注意力热力图已保存: {save_path}")

    plt.show()


print("注意力可视化函数定义完成！")"""

code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_cell_1_source
}
new_cells.append(code_cell_1)

# Cell 3: Example visualization
code_cell_2_source = """# --- 7.4.1 选择示例并可视化注意力 ---
print("=" * 60)
print("Seq2Seq 注意力权重可视化示例")
print("=" * 60)

# 选择一个测试样本 (index=42)
example_idx = 42
src_tensor = torch.tensor(eng_test[example_idx], dtype=torch.long)

# 翻译并获取注意力权重
translated_ids, attention_matrix = translate_with_attention(model_seq2seq, src_tensor)

# 将token ids转换为单词
src_words = []
for idx in eng_test[example_idx]:
    if idx == PAD_IDX:
        break
    word = eng_vocab.idx2word.get(idx, '<unk>')
    if word not in ['<sos>', '<eos>']:
        src_words.append(word)

tgt_words = [spa_vocab.idx2word.get(idx, '<unk>') for idx in translated_ids]
ref_words = []
for idx in spa_test[example_idx]:
    if idx == PAD_IDX:
        break
    word = spa_vocab.idx2word.get(idx, '<unk>')
    if word not in ['<sos>', '<eos>']:
        ref_words.append(word)

# 打印翻译结果
print(f"\\n源句子 (英语):   {' '.join(src_words)}")
print(f"参考翻译 (西班牙语): {' '.join(ref_words)}")
print(f"模型翻译 (西班牙语): {' '.join(tgt_words)}")
print(f"\\n注意力矩阵形状: {attention_matrix.shape}")
print()

# 可视化注意力权重
if len(tgt_words) > 0 and len(src_words) > 0:
    visualize_attention(
        input_words=src_words,
        output_words=tgt_words,
        attention_weights=attention_matrix,
        save_path='attention_seq2seq_example.png'
    )
else:
    print("警告: 翻译结果为空，无法可视化注意力权重")"""

code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_cell_2_source
}
new_cells.append(code_cell_2)

# Cell 4: Multiple examples
code_cell_3_source = """# --- 7.4.2 可视化多个示例的注意力 ---
print("\\n" + "=" * 60)
print("可视化更多注意力示例")
print("=" * 60)

# 选择3个不同长度的示例
attention_examples = [10, 25, 50]  # 不同的测试样本索引

for i, example_idx in enumerate(attention_examples, 1):
    if example_idx >= len(eng_test):
        continue

    src_tensor = torch.tensor(eng_test[example_idx], dtype=torch.long)
    translated_ids, attention_matrix = translate_with_attention(model_seq2seq, src_tensor)

    # 转换为单词
    src_words = []
    for idx in eng_test[example_idx]:
        if idx == PAD_IDX:
            break
        word = eng_vocab.idx2word.get(idx, '<unk>')
        if word not in ['<sos>', '<eos>']:
            src_words.append(word)

    tgt_words = [spa_vocab.idx2word.get(idx, '<unk>') for idx in translated_ids]

    print(f"\\n[示例 {i}]")
    print(f"源句子: {' '.join(src_words)}")
    print(f"翻译:   {' '.join(tgt_words)}")

    if len(tgt_words) > 0 and len(src_words) > 0:
        visualize_attention(
            input_words=src_words,
            output_words=tgt_words,
            attention_weights=attention_matrix,
            save_path=f'attention_seq2seq_example_{i}.png'
        )

print("\\n" + "=" * 60)
print("注意力可视化完成！")
print("已保存图片:")
print("  - attention_seq2seq_example.png")
print("  - attention_seq2seq_example_1.png")
print("  - attention_seq2seq_example_2.png")
print("  - attention_seq2seq_example_3.png")
print("=" * 60)"""

code_cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_cell_3_source
}
new_cells.append(code_cell_3)

# Insert the new cells before section 8 (index 21)
nb['cells'] = nb['cells'][:21] + new_cells + nb['cells'][21:]

# Save the modified notebook
with open('machine_translation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Successfully added attention visualization cells to the notebook!")
print(f"Added {len(new_cells)} new cells before section 8")
