# 误差分析报告 (Error Analysis Report)

> 数据来源: 三个任务 `outputs/` 目录下完整训练后的真实输出
> 生成日期: 2026-04-29
> 仓库内相对路径作为引用，便于跨机器复现

---

## 1. IMDB 情感二分类

### 1.1 整体误差汇总

- 数据来源: `情感二分类/outputs/sentiment_error_summary.csv`
- 最优模型: **NaiveBayes (TF-IDF + 多项式 NB)**
- 测试集规模: 5000 条

| 类别 | 计数 | 占比 |
|------|-----:|-----:|
| 正确预测 (correct) | 4315 | 86.30 % |
| 假阴性 (false_negative) | 364 | 7.28 % |
| 假阳性 (false_positive) | 321 | 6.42 % |
| **错误合计** | **685** | **13.70 %** |

```text
best_model,error_type,count
NaiveBayes,correct,4315
NaiveBayes,false_negative,364
NaiveBayes,false_positive,321
```

### 1.2 误差案例采样

- 数据来源: `情感二分类/outputs/sentiment_error_analysis.csv`
- 采样规模: 400 条（按预测概率排序，FN 213 条 / FP 187 条）

观察: 错误样本的预测概率多在 0.9 以上仍判错，说明 NaiveBayes 在面对反讽、混合情感、长文本中部分情感词主导整段倾向等情况时仍会"高置信度误判"。
后续优化建议: 在论文 4.1 节挑 3–5 个典型 case 做语义分析，结合 TF-IDF 词权重解释为什么模型被特定关键词误导。

---

## 2. Reuters-46 新闻多分类

### 2.1 各模型 Top 混淆对

- 数据来源: `新闻多分类/outputs/reuters_top_confusions.csv`
- 测试集规模: 2246 条 (46 类，长尾严重)

各模型最频繁的 5 组混淆 (按 count 降序):

| 模型 | Top-5 混淆对 (true → pred : count) |
|------|------------------------------------|
| **TextCNN** (最优 Macro-F1=0.6050) | 4 → 3: 31; 3 → 4: 21; 20 → 19: 11; 4 → 16: 10; 13 → 16: 10 |
| **BiGRU** (Macro-F1=0.4881) | 3 → 4: 23; 4 → 3: 20; 20 → 19: 16; 4 → 16: 15; 16 → 4: 9 |
| **Transformer** (Macro-F1=0.4815) | 4 → 3: 24; 3 → 4: 23; 19 → 20: 18; 16 → 4: 9; 13 → 16: 8 |
| **NaiveBayes** (Macro-F1=0.5584) | 3 → 4: 66; 19 → 20: 18; 11 → 36: 14; 3 → 20: 11; 20 → 19: 11 |

### 2.2 共性观察

- **类别 3 ↔ 类别 4 双向混淆**普遍存在于所有四种模型。这两个类样本量最大 (3: 795, 4: 485)，且主题相近，说明问题来自标签语义重叠而非模型能力。
- **类别 19 ↔ 类别 20 同样高频互判**，三个深度模型均出现，是次要的相邻主题问题。
- 类别 16 (109 条) 频繁吸收来自 4、13、20 的样本，可能因 16 的关键词分布范围广、特征边界较模糊。
- NaiveBayes 在 3 → 4 上的混淆数量 (66) 远高于深度模型 (21–24)，说明朴素贝叶斯对相邻语义类的分辨力最弱，但其对长尾类的覆盖反而比 BiGRU/Transformer 略好。

### 2.3 优化建议 (对接中期"定向优化"阶段)

1. 对类别 3、4 引入类别加权或 focal loss，降低占比惩罚
2. 检查 reuters 原始标签定义，确认 3↔4 是否本身就是高度相关主题，必要时合并或分层分类
3. 提升序列长度上限以保留更多上下文区分相近类
4. 用 Macro-F1 而非 Accuracy 选 best epoch（当前 TextCNN best_val_f1_macro=0.5809 但测试 0.6050，提示 early stop 触发偏早）

---

## 3. 西班牙语 → 英语 机器翻译

### 3.1 错误类型汇总

- 数据来源: `机器翻译/outputs/translation_error_summary.csv`
- 错误样本采样规模: 320 条 (每模型/解码组合各采 80 条进行人工启发式标注)
- 评估测试集规模: 3200 条 (BLEU 报告基于此)

| 模型 - 解码 | semantic_or_grammar_error | under_translation | over_translation | low_keyword_overlap |
|-------------|--------------------------:|------------------:|-----------------:|--------------------:|
| seq2seq_greedy | 80 | 0 | 0 | 0 |
| seq2seq_beam | 79 | 0 | 0 | 1 |
| transformer_greedy | 72 | 6 | 1 | 1 |
| transformer_beam | 71 | 8 | 0 | 1 |

### 3.2 错误自动化标注规则 (来自 `translation_error_analysis.csv`)

- 列: `source_spanish, reference_english, model_decode, prediction, token_overlap, length_ratio, error_tag`
- 平均 token_overlap = 0.668，平均 length_ratio = 0.955
- `under_translation`: length_ratio < 0.7 (输出明显短于参考)
- `over_translation`: length_ratio > 1.3
- `low_keyword_overlap`: token_overlap < 0.3
- 其余被标为 `semantic_or_grammar_error` (语义/语法错误)

### 3.3 模型差异观察

- **Seq2Seq+Attention**: 错误几乎全部归类为 semantic_or_grammar_error，没有显著漏译/超译，说明长度控制良好，但语义和语法表达欠精确。
- **Transformer**: 出现 6–8 例 under_translation，提示其在长句或复合句下出现"提前终止 / 漏译尾段"现象。**这是 Transformer BLEU-4 (0.318) 落后于 Seq2Seq (0.349) 的主要原因之一**。
- Beam Search 相比 Greedy 在两个模型上都让 BLEU-4 提升 ~0.015–0.018，但 Seq2Seq 推理速度从 62 句/s 跌到 15 句/s，Transformer 从 11 句/s 跌到 3 句/s。

### 3.4 优化建议

1. 对 Transformer 调整 length penalty / coverage penalty，缓解漏译
2. 训练时引入 label smoothing 0.1
3. Beam size 网格搜索 (1/3/5/8) + α length penalty (0.6–1.2)
4. 平行语料二次清洗：低频词、长度差异 > 2x 的句对可剔除

---

## 4. 后续工作衔接

本报告基于完整训练后的真实输出生成，可直接对应论文 4.x 节"误差分析"的素材；图形化版本见:

- `figures/fig_sentiment_confusion.png` (情感任务 2×2 混淆矩阵)
- `figures/fig_sentiment_error_breakdown.png` (情感任务误差类型饼图/柱图)
- `figures/fig_reuters_top_confusions_textcnn.png` 与 `fig_reuters_top_confusions_bigru.png` (Reuters 混淆热图)
- `figures/fig_reuters_per_class_f1.png` (Reuters 各类 F1 与样本量长尾)
- `figures/fig_translation_error_distribution.png` (翻译错误类型堆叠柱图)

---

> **历史版本说明**: 旧 `error_analysis_report.md` 引用的是 `outputs_smoke/`(快速验证模式) 下的小规模数据，且包含本机以外的 Windows 绝对路径。本版本已替换为完整训练 outputs 的真实数据。
