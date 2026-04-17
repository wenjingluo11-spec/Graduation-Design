机器翻译模块说明（2026-03-31 更新）
================================

本模块已重构为“可完整跑通”的基础模型实验版本，符合开题报告要求：
- 任务：西班牙语 -> 英语 机器翻译
- 模型：Seq2Seq + Bahdanau Attention、Transformer
- 指标：BLEU-1 / BLEU-2 / BLEU-4

主入口
------
1) Python 脚本：machine_translation.py
2) Notebook：machine_translation.ipynb（调用主脚本）

运行方式
------
快速验证（推荐先跑）：
python machine_translation.py --quick

完整实验：
python machine_translation.py

输出文件
------
默认输出到本目录 outputs/：
- translation_results.csv
- translation_config.json
- training_curves.png
- translation_examples.csv

设计要点
------
1) 不再强制 CUDA；自动 CPU/GPU 兼容
2) 数据路径自动解析（支持从模块目录或项目根目录运行）
3) train/val/test 明确划分，词表仅基于训练集构建，避免数据泄漏
4) 训练包含早停，评估包含 Beam Search 与 BLEU 统计
