# 修复 test_pairs 未定义问题

## 问题原因

新增的 7.4-7.8 部分代码需要使用 `test_pairs` 变量（原始文本对格式），但在数据处理后，数据被转换成了张量格式（`eng_test`, `spa_test`），原始的文本对格式丢失了。

## 解决方案

在 **7.4 第一个单元格之前**（即 `def translate_with_attention` 函数定义之前）插入以下新单元格：

---

### 新增单元格：重建 test_pairs

```python
# ============================================================================
# 7.3.5 重建测试集文本对（供后续分析使用）
# ============================================================================

# 从张量格式重建原始文本对
test_pairs = []

for i in range(len(eng_test)):
    # 解码英语句子
    eng_ids = [idx for idx in eng_test[i] if idx not in (PAD_IDX, SOS_IDX, EOS_IDX)]
    eng_words = [eng_vocab.idx2word[idx] for idx in eng_ids]
    eng_sentence = ' '.join(eng_words)

    # 解码西班牙语句子
    spa_ids = [idx for idx in spa_test[i] if idx not in (PAD_IDX, SOS_IDX, EOS_IDX)]
    spa_words = [spa_vocab.idx2word[idx] for idx in spa_ids]
    spa_sentence = ' '.join(spa_words)

    test_pairs.append((eng_sentence, spa_sentence))

print(f"已重建 test_pairs: {len(test_pairs)} 对句子")
print(f"示例: {test_pairs[0]}")
```

---

## 插入位置

在你的 notebook 中找到这个单元格：

```python
def translate_with_attention(model, src_sentence, src_vocab, tgt_vocab, device, max_len=50):
    """翻译并返回注意力权重"""
    ...
```

在这个单元格**之前**插入上面的新单元格。

## 验证

插入后，按顺序运行：
1. 运行新插入的单元格（重建 test_pairs）
2. 运行 7.4 第二个单元格（注意力可视化示例）

应该不会再出现 `NameError: name 'test_pairs' is not defined` 错误。

## 注意事项

- 这个单元格需要在运行 7.4-7.8 的任何代码之前运行
- 如果你重新运行了前面的数据处理部分，需要重新运行这个单元格
- `eng_vocab` 和 `spa_vocab` 必须已经定义（在前面的单元格中）
