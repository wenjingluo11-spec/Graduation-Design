import json

with open('machine_translation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 找到 Section 8
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '第8部分' in source:
            insert_idx = i
            break

if not insert_idx:
    print("Error: Section 8 not found")
    exit(1)

print(f"Insert at: {insert_idx}, Total cells: {len(nb['cells'])}")

# 简单的创建函数
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split('\n')}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.split('\n')}

new = []

# Section 7.7
new.append(md("## 7.7 统计显著性测试\n\n使用配对 t 检验验证两个模型之间的性能差异是否具有统计显著性。"))

new.append(code("""from scipy import stats

def paired_t_test(model1_scores, model2_scores, model1_name, model2_name):
    if len(model1_scores) != len(model2_scores):
        raise ValueError("分数列表长度不匹配")
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    differences = np.array(model1_scores) - np.array(model2_scores)
    mean_diff = differences.mean()
    return {'t_statistic': t_stat, 'p_value': p_value, 'significant': p_value < 0.05,
            'mean_diff': mean_diff, 'model1_name': model1_name, 'model2_name': model2_name}

def compute_per_sample_bleu(model, test_pairs, src_vocab, tgt_vocab, device, translate_fn):
    bleu_scores = []
    for i, (src, tgt) in enumerate(test_pairs, 1):
        try:
            hyp = translate_fn(model, src, src_vocab, tgt_vocab, device)
            bleu_score = compute_bleu_scores([[tgt.split()]], [hyp.split()])
            bleu_scores.append(bleu_score['bleu_4'])
        except:
            bleu_scores.append(0.0)
        if i % 100 == 0:
            print(f"  已处理 {i}/{len(test_pairs)}")
    return bleu_scores

print("统计测试函数已定义")"""))

new.append(code("""print("计算 Seq2Seq BLEU 分数")
seq2seq_scores = compute_per_sample_bleu(seq2seq_model, test_pairs[:200], eng_vocab, spa_vocab, DEVICE, translate_sentence)
print(f"Seq2Seq 平均: {np.mean(seq2seq_scores):.4f}")"""))

new.append(code("""print("计算 Transformer BLEU 分数")
transformer_scores = compute_per_sample_bleu(transformer_model, test_pairs[:200], eng_vocab, spa_vocab, DEVICE, translate_sentence_tf)
print(f"Transformer 平均: {np.mean(transformer_scores):.4f}")"""))

new.append(code("""result = paired_t_test(transformer_scores, seq2seq_scores, 'Transformer', 'Seq2Seq')
print(f"\nt 统计量: {result['t_statistic']:.4f}")
print(f"p 值: {result['p_value']:.6f}")
print(f"显著性: {'是' if result['significant'] else '否'}")"""))

# Section 7.8
new.append(md("## 7.8 交互式翻译演示\n\n提供批量翻译演示，方便答辩展示。"))

new.append(code("""def batch_translation_demo(sentences, seq2seq_model, transformer_model, src_vocab, tgt_vocab, device):
    results = []
    for eng, spa in sentences:
        seq2seq_trans = translate_sentence(seq2seq_model, eng, src_vocab, tgt_vocab, device)
        tf_trans = translate_sentence_tf(transformer_model, eng, src_vocab, tgt_vocab, device)
        results.append({'source': eng, 'reference': spa, 'seq2seq': seq2seq_trans, 'transformer': tf_trans})
    return results

print("演示函数已定义")"""))

new.append(code("""demo_sentences = [
    ("i love you", "te quiero"),
    ("how are you", "como estas"),
    ("good morning", "buenos dias"),
]

print("批量翻译演示")
demo_results = batch_translation_demo(demo_sentences, seq2seq_model, transformer_model, eng_vocab, spa_vocab, DEVICE)

for i, r in enumerate(demo_results, 1):
    print(f"\n示例 {i}")
    print(f"源句子: {r['source']}")
    print(f"参考: {r['reference']}")
    print(f"Seq2Seq: {r['seq2seq']}")
    print(f"Transformer: {r['transformer']}")"""))

nb['cells'] = nb['cells'][:insert_idx] + new + nb['cells'][insert_idx:]

with open('machine_translation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done! Added {len(new)} cells. Total: {len(nb['cells'])}")
