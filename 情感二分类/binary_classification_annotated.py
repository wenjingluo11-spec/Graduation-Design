"""
情感二分类实验代码 (带详细注释)
=====================================
基于 IMDB 数据集的情感分析，包含以下模型：
1. Naive Bayes (传统机器学习基准)
2. TextCNN (卷积神经网络)
3. BiLSTM (双向循环神经网络)
4. BERT (预训练模型微调)

Author: Auto-generated with annotations
"""

# =====================================================================
# 第1部分：导入依赖库
# =====================================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# NLTK: 自然语言处理工具包
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn: 传统机器学习
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# PyTorch: 深度学习框架
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Transformers: Hugging Face 预训练模型库
from transformers import BertTokenizer, BertForSequenceClassification

# 下载 NLTK 资源
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


# =====================================================================
# 第2部分：全局配置参数
# =====================================================================
MAX_VOCAB_SIZE = 10000   # 词表最大大小，只保留最常见的10000个词
MAX_SEQ_LEN = 200        # 输入序列最大长度，超过则截断，不足则填充
EMBEDDING_DIM = 128      # 词向量维度，每个词用128维向量表示
BATCH_SIZE = 64          # 每批训练的样本数量
EPOCHS = 5               # 训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')


# =====================================================================
# 第3部分：数据加载与预处理
# =====================================================================
def load_imdb_data(data_dir='aclImdb'):
    """
    加载 IMDB 电影评论数据集
    
    数据集结构:
    aclImdb/
    ├── train/
    │   ├── pos/   (正面评论, 标签=1)
    │   └── neg/   (负面评论, 标签=0)
    └── test/
        ├── pos/
        └── neg/
    
    Args:
        data_dir: 数据集根目录
    
    Returns:
        train_df, test_df: 训练集和测试集 DataFrame
    """
    def load_data_from_dir(dir_path, label):
        """从目录读取所有 .txt 文件"""
        texts, labels = [], []
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
        return texts, labels
    
    # 分别加载训练集和测试集的正负样本
    train_pos, train_pos_labels = load_data_from_dir(os.path.join(data_dir, 'train', 'pos'), 1)
    train_neg, train_neg_labels = load_data_from_dir(os.path.join(data_dir, 'train', 'neg'), 0)
    test_pos, test_pos_labels = load_data_from_dir(os.path.join(data_dir, 'test', 'pos'), 1)
    test_neg, test_neg_labels = load_data_from_dir(os.path.join(data_dir, 'test', 'neg'), 0)
    
    # 合并为 DataFrame
    train_df = pd.DataFrame({
        'text': train_pos + train_neg,
        'label': train_pos_labels + train_neg_labels
    })
    test_df = pd.DataFrame({
        'text': test_pos + test_neg,
        'label': test_pos_labels + test_neg_labels
    })
    
    # 打乱顺序，确保正负样本随机分布
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train_df, test_df


def clean_text(text):
    """
    文本清洗流程
    
    处理步骤:
    1. 去除 HTML 标签 (如 <br>, <p> 等)
    2. 只保留英文字母和空格
    3. 转换为小写
    4. 使用 NLTK 分词
    5. 去除停用词 (the, is, a, an 等无实际意义的词)
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    text = re.sub(r'<.*?>', '', text)           # 去除 HTML 标签
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # 只保留字母和空格
    text = text.lower()                          # 转小写
    tokens = word_tokenize(text)                 # NLTK 分词
    tokens = [t for t in tokens if t not in stop_words]  # 去停用词
    return ' '.join(tokens)


# 加载并清洗数据
print("加载数据...")
train_df, test_df = load_imdb_data()
print("清洗文本...")
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
y_train = train_df['label'].values
y_test = test_df['label'].values
print(f"训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")


# =====================================================================
# 第4部分：朴素贝叶斯基准模型 (Naive Bayes Baseline)
# =====================================================================
"""
朴素贝叶斯分类器原理:
- 基于贝叶斯定理: P(类别|特征) ∝ P(特征|类别) × P(类别)
- "朴素"假设: 各特征之间相互独立
- 适用于文本分类，计算速度快，效果不错

TF-IDF (词频-逆文档频率):
- TF (词频): 词在文档中出现的频率
- IDF (逆文档频率): log(总文档数 / 包含该词的文档数)
- TF-IDF = TF × IDF
- 作用: 衡量词对文档的重要程度，过滤掉常见但无意义的词
"""
print("\n===== 训练朴素贝叶斯模型 =====")
tfidf = TfidfVectorizer(max_features=MAX_VOCAB_SIZE)
X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])  # 拟合并转换训练集
X_test_tfidf = tfidf.transform(test_df['clean_text'])        # 只转换测试集

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

nb_acc = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
print(f'Naive Bayes - 准确率: {nb_acc:.4f}, F1分数: {nb_f1:.4f}')


# =====================================================================
# 第5部分：构建词表 (Vocabulary Building)
# =====================================================================
"""
深度学习模型需要将文本转为数字序列:
文本 -> 分词 -> 查词表 -> 整数序列 -> Embedding 向量

词表结构:
- <pad>: 填充符，索引=0，用于将短序列填充到固定长度
- <unk>: 未知词，索引=1，用于表示词表外的词
- 其他词: 按词频排序
"""
print("\n===== 构建词表 =====")

def build_vocab(texts, max_tokens):
    """
    从文本语料构建词表
    
    Args:
        texts: 文本列表
        max_tokens: 词表最大大小
    
    Returns:
        word2idx: 词到索引的映射字典
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    # 保留最常见的词
    most_common = counter.most_common(max_tokens - 2)  # 预留 <pad> 和 <unk>
    
    word2idx = {'<pad>': 0, '<unk>': 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    
    return word2idx

vocab = build_vocab(train_df['clean_text'], MAX_VOCAB_SIZE)
PAD_IDX = vocab['<pad>']
UNK_IDX = vocab['<unk>']
print(f"词表大小: {len(vocab)}")


def encode(text):
    """
    将文本编码为整数序列
    
    处理流程:
    1. 分词
    2. 查词表获取索引 (未知词用 <unk>)
    3. 填充/截断到固定长度
    
    Args:
        text: 清洗后的文本
    
    Returns:
        ids: 整数索引列表
    """
    tokens = text.split()
    ids = [vocab.get(token, UNK_IDX) for token in tokens]
    
    if len(ids) < MAX_SEQ_LEN:
        ids = ids + [PAD_IDX] * (MAX_SEQ_LEN - len(ids))  # 填充
    else:
        ids = ids[:MAX_SEQ_LEN]  # 截断
    
    return ids


# 编码所有数据
print("编码文本...")
train_ids = np.array([encode(t) for t in train_df['clean_text']])
test_ids = np.array([encode(t) for t in test_df['clean_text']])

# 创建 PyTorch Dataset 和 DataLoader
train_dataset = data.TensorDataset(
    torch.tensor(train_ids, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = data.TensorDataset(
    torch.tensor(test_ids, dtype=torch.long),
    torch.tensor(y_test, dtype=torch.long)
)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


# =====================================================================
# 第6部分：TextCNN 模型
# =====================================================================
"""
TextCNN 核心思想:
- 使用卷积神经网络提取文本的局部特征 (n-gram)
- kernel_size=3 的卷积核可以捕捉 3 个连续词的组合特征

网络结构:
Input -> Embedding -> Conv1D -> ReLU -> MaxPool -> Conv1D -> GlobalMaxPool -> Dropout -> FC -> Sigmoid

优点: 并行计算，训练速度快
缺点: 难以捕捉长距离依赖关系
"""
print("\n===== 训练 TextCNN 模型 =====")

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=1):
        super().__init__()
        # Embedding 层: 将词索引映射为稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        
        # 卷积层: 提取局部 n-gram 特征
        # kernel_size=3: 每次看 3 个连续的词
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 池化层: 降维
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        
        # 全连接层
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout: 防止过拟合
        self.sigmoid = nn.Sigmoid()      # Sigmoid: 输出概率

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len) 整数索引
        
        Returns:
            (batch_size,) 预测概率
        """
        # Embedding: (batch, seq) -> (batch, seq, embed)
        x = self.embedding(x)
        
        # 转置: (batch, seq, embed) -> (batch, embed, seq)
        # Conv1d 期望输入格式为 (batch, channels, length)
        x = x.permute(0, 2, 1)
        
        # 第一个卷积块
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        
        # 第二个卷积块
        x = torch.relu(self.conv2(x))
        
        # 全局最大池化: 取每个通道的最大值
        # (batch, 128, length) -> (batch, 128)
        x = torch.max(x, dim=2)[0]
        
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(1)


# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.float().to(DEVICE)
        
        optimizer.zero_grad()       # 清零梯度
        preds = model(xb)           # 前向传播
        loss = criterion(preds, yb) # 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数
        
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(model, loader):
    """评估模型"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    
    # 将概率转为类别标签 (阈值=0.5)
    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)
    return acc, f1


# 初始化模型
model_cnn = TextCNN(len(vocab), EMBEDDING_DIM).to(DEVICE)
criterion_cnn = nn.BCELoss()  # 二元交叉熵损失
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=1e-3)

# 训练循环
for epoch in range(EPOCHS):
    loss = train_epoch(model_cnn, train_loader, criterion_cnn, optimizer_cnn)
    acc, f1 = eval_model(model_cnn, test_loader)
    print(f'TextCNN Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - 准确率: {acc:.4f} - F1: {f1:.4f}')

cnn_acc, cnn_f1 = eval_model(model_cnn, test_loader)


# =====================================================================
# 第7部分：BiLSTM 模型
# =====================================================================
"""
BiLSTM (双向长短期记忆网络) 核心思想:
- LSTM: 通过门控机制解决 RNN 的梯度消失问题，能捕捉长距离依赖
- 双向: 同时从左到右和从右到左阅读文本，获取更完整的上下文信息

网络结构:
Input -> Embedding -> BiLSTM -> Dropout -> FC -> Sigmoid

优点: 能捕捉长距离依赖和上下文信息
缺点: 序列处理，训练速度较慢
"""
print("\n===== 训练 BiLSTM 模型 =====")

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        
        # LSTM 层
        # hidden_dim: 隐状态维度
        # bidirectional=True: 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # 全连接层 (hidden_dim * 2 因为是双向)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        LSTM 输出:
        - output: (batch, seq, hidden*2) 每个时间步的隐状态
        - (h_n, c_n): 最后时间步的隐状态和记忆状态
        """
        embed = self.embedding(x)  # (batch, seq, embed)
        
        # LSTM 处理
        lstm_out, _ = self.lstm(embed)  # (batch, seq, hidden*2)
        
        # 取最后一个时间步的隐状态作为句子表示
        out = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)


# 初始化模型
model_lstm = BiLSTM(len(vocab), EMBEDDING_DIM).to(DEVICE)
criterion_lstm = nn.BCELoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=1e-3)

# 训练循环
for epoch in range(EPOCHS):
    loss = train_epoch(model_lstm, train_loader, criterion_lstm, optimizer_lstm)
    acc, f1 = eval_model(model_lstm, test_loader)
    print(f'BiLSTM Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - 准确率: {acc:.4f} - F1: {f1:.4f}')

lstm_acc, lstm_f1 = eval_model(model_lstm, test_loader)


# =====================================================================
# 第8部分：BERT 微调
# =====================================================================
"""
BERT (Bidirectional Encoder Representations from Transformers) 核心思想:
- 预训练: 在大规模语料上预训练，学习通用的语言表示
- 微调: 在特定任务上微调，只需少量数据即可获得很好效果

BERT 特点:
- 使用 Transformer Encoder 架构
- 双向上下文建模 (不同于 GPT 的单向)
- WordPiece 子词分词 (处理未登录词)

微调策略:
- 使用很小的学习率 (2e-5)，避免破坏预训练权重
- 通常只需 2-4 个 epoch
"""
print("\n===== 训练 BERT 模型 =====")

# 加载预训练的 BERT Tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')


def encode_bert(texts, max_len=128):
    """
    使用 BERT Tokenizer 编码文本
    
    BERT 输入需要:
    - input_ids: token 索引
    - attention_mask: 1=真实token, 0=padding
    """
    encodings = tokenizer_bert(
        texts.tolist(),
        truncation=True,        # 超长截断
        padding='max_length',   # 填充到 max_length
        max_length=max_len,
        return_tensors='pt'     # 返回 PyTorch 张量
    )
    return encodings['input_ids'], encodings['attention_mask']


# 使用子集进行演示 (BERT 训练较慢)
train_texts_bert = train_df['clean_text'][:5000]
test_texts_bert = test_df['clean_text'][:2000]
train_labels_bert = torch.tensor(train_df['label'].values[:5000], dtype=torch.long)
test_labels_bert = torch.tensor(test_df['label'].values[:2000], dtype=torch.long)

print("BERT 编码中...")
train_ids_bert, train_masks = encode_bert(train_texts_bert)
test_ids_bert, test_masks = encode_bert(test_texts_bert)

# 创建 DataLoader
train_dataset_bert = data.TensorDataset(train_ids_bert, train_masks, train_labels_bert)
test_dataset_bert = data.TensorDataset(test_ids_bert, test_masks, test_labels_bert)
train_loader_bert = data.DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
test_loader_bert = data.DataLoader(test_dataset_bert, batch_size=16)

# 加载预训练 BERT + 分类头
model_bert = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # 二分类
).to(DEVICE)

# AdamW: 带权重衰减的 Adam 优化器
optimizer_bert = optim.AdamW(model_bert.parameters(), lr=2e-5)


def train_bert_epoch(epoch):
    """训练 BERT 一个 epoch"""
    model_bert.train()
    total_loss = 0
    for ids, masks, labels in train_loader_bert:
        ids = ids.to(DEVICE)
        masks = masks.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer_bert.zero_grad()
        
        # BERT 前向传播 (传入 labels 会自动计算损失)
        outputs = model_bert(ids, attention_mask=masks, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer_bert.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader_bert)


def eval_bert():
    """评估 BERT 模型"""
    model_bert.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, masks, labels in test_loader_bert:
            ids = ids.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model_bert(ids, attention_mask=masks)
            logits = outputs.logits  # (batch, 2)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1


# 训练循环 (BERT 通常只需 2-3 个 epoch)
for epoch in range(2):
    loss = train_bert_epoch(epoch)
    acc, f1 = eval_bert()
    print(f'BERT Epoch {epoch+1}/2 - Loss: {loss:.4f} - 准确率: {acc:.4f} - F1: {f1:.4f}')

bert_acc, bert_f1 = eval_bert()


# =====================================================================
# 第9部分：结果汇总与可视化
# =====================================================================
print("\n===== 实验结果汇总 =====")

results = pd.DataFrame({
    '模型': ['Naive Bayes', 'TextCNN', 'BiLSTM', 'BERT'],
    '准确率': [nb_acc, cnn_acc, lstm_acc, bert_acc],
    'F1分数': [nb_f1, cnn_f1, lstm_f1, bert_f1]
})
print(results)

# 绘制对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 准确率对比
axes[0].bar(results['模型'], results['准确率'], color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
axes[0].set_title('模型准确率对比', fontsize=14)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Accuracy')

# F1分数对比
axes[1].bar(results['模型'], results['F1分数'], color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
axes[1].set_title('模型F1分数对比', fontsize=14)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('F1 Score')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

print("\n实验完成!")
