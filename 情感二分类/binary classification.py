import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, Dropout
from transformers import BertTokenizer, TFBertForSequenceClassification

# 下载NLTK必要资源
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


# ===================== 1. 数据加载与预处理 =====================
def load_imdb_data(data_dir='aclImdb'):
    """
    加载aclImdb数据集（需先从http://ai.stanford.edu/~amaas/data/sentiment/下载解压）
    返回：DataFrame格式的训练集、测试集
    """

    # 定义加载单类数据的函数
    def load_data_from_dir(dir_path, label):
        texts = []
        labels = []
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
        return texts, labels

    # 加载训练集（pos=1, neg=0）
    train_pos_texts, train_pos_labels = load_data_from_dir(os.path.join(data_dir, 'train', 'pos'), 1)
    train_neg_texts, train_neg_labels = load_data_from_dir(os.path.join(data_dir, 'train', 'neg'), 0)
    train_texts = train_pos_texts + train_neg_texts
    train_labels = train_pos_labels + train_neg_labels

    # 加载测试集
    test_pos_texts, test_pos_labels = load_data_from_dir(os.path.join(data_dir, 'test', 'pos'), 1)
    test_neg_texts, test_neg_labels = load_data_from_dir(os.path.join(data_dir, 'test', 'neg'), 0)
    test_texts = test_pos_texts + test_neg_texts
    test_labels = test_pos_labels + test_neg_labels

    # 转为DataFrame
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

    # 打乱数据
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, test_df


def clean_text(text):
    """文本清洗：去特殊字符、小写、去停用词"""
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转为小写
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 去停用词
    tokens = [token for token in tokens if token not in stop_words]
    # 拼接回文本
    clean_text = ' '.join(tokens)
    return clean_text


# 加载并清洗数据
print("加载aclImdb数据集...")
train_df, test_df = load_imdb_data()
print("文本清洗中...")
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

# ===================== 2. 基线模型1：朴素贝叶斯（TF-IDF） =====================
print("\n===== 训练朴素贝叶斯模型 =====")
# TF-IDF向量化
tfidf = TfidfVectorizer(max_features=10000)  # 保留Top10000特征
X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
X_test_tfidf = tfidf.transform(test_df['clean_text'])
y_train = train_df['label'].values
y_test = test_df['label'].values

# 训练模型
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 预测与评估
y_pred_nb = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
print(f"朴素贝叶斯 - 准确率: {nb_acc:.4f}, F1-score: {nb_f1:.4f}")

# ===================== 3. 基线模型2：TextCNN（词嵌入） =====================
print("\n===== 训练TextCNN模型 =====")
# 文本序列化
max_vocab_size = 10000
max_seq_len = 200  # 统一文本长度
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(train_df['clean_text'])

# 转为序列并填充
X_train_seq = tokenizer.texts_to_sequences(train_df['clean_text'])
X_test_seq = tokenizer.texts_to_sequences(test_df['clean_text'])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_len, padding='post', truncating='post')

# 构建TextCNN模型
embedding_dim = 128
cnn_model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # 防止过拟合
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# 训练模型
history_cnn = cnn_model.fit(
    X_train_pad, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# 评估模型
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_pad, y_test, verbose=0)
y_pred_cnn = (cnn_model.predict(X_test_pad) > 0.5).astype(int).flatten()
cnn_f1 = f1_score(y_test, y_pred_cnn)
print(f"TextCNN - 准确率: {cnn_acc:.4f}, F1-score: {cnn_f1:.4f}")

# ===================== 4. 基线模型3：Bi-LSTM =====================
print("\n===== 训练Bi-LSTM模型 =====")
# 构建Bi-LSTM模型
lstm_model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

# 训练模型
history_lstm = lstm_model.fit(
    X_train_pad, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# 评估模型
lstm_loss, lstm_acc = lstm_model.evaluate(X_test_pad, y_test, verbose=0)
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype(int).flatten()
lstm_f1 = f1_score(y_test, y_pred_lstm)
print(f"Bi-LSTM - 准确率: {lstm_acc:.4f}, F1-score: {lstm_f1:.4f}")

# ===================== 5. 进阶模型：BERT微调 =====================
print("\n===== 训练BERT模型 =====")
# 加载BERT分词器和模型
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)


# BERT文本编码
def encode_texts(texts, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)


# 编码训练/测试集
X_train_ids, X_train_masks = encode_texts(train_df['clean_text'][:10000], bert_tokenizer)  # 缩减样本加速训练
X_test_ids, X_test_masks = encode_texts(test_df['clean_text'][:2000], bert_tokenizer)
y_train_bert = y_train[:10000]
y_test_bert = y_test[:2000]

# 构建BERT训练模型
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_masks = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_masks')
bert_output = bert_model([input_ids, attention_masks])[0]
output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)

bert_train_model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
bert_train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

# 训练BERT（仅1轮，加速演示）
history_bert = bert_train_model.fit(
    [X_train_ids, X_train_masks], y_train_bert,
    batch_size=16,
    epochs=1,
    validation_split=0.1,
    verbose=1
)

# 评估BERT
bert_loss, bert_acc = bert_train_model.evaluate([X_test_ids, X_test_masks], y_test_bert, verbose=0)
y_pred_bert = (bert_train_model.predict([X_test_ids, X_test_masks]) > 0.5).astype(int).flatten()
bert_f1 = f1_score(y_test_bert, y_pred_bert)
print(f"BERT - 准确率: {bert_acc:.4f}, F1-score: {bert_f1:.4f}")

# ===================== 6. 实验结果可视化与对比 =====================
print("\n===== 实验结果对比 =====")
# 整理结果
results = pd.DataFrame({
    '模型': ['朴素贝叶斯', 'TextCNN', 'Bi-LSTM', 'BERT'],
    '准确率': [nb_acc, cnn_acc, lstm_acc, bert_acc],
    'F1-score': [nb_f1, cnn_f1, lstm_f1, bert_f1]
})
print(results)

# 1. 准确率对比图
plt.figure(figsize=(10, 5))
sns.barplot(x='模型', y='准确率', data=results)
plt.title('各模型准确率对比')
plt.ylim(0, 1)
plt.show()

# 2. Bi-LSTM混淆矩阵（示例）
cm = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('Bi-LSTM混淆矩阵')
plt.show()

# 3. TextCNN训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='训练准确率')
plt.plot(history_cnn.history['val_accuracy'], label='验证准确率')
plt.title('TextCNN准确率曲线')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='训练损失')
plt.plot(history_cnn.history['val_loss'], label='验证损失')
plt.title('TextCNN损失曲线')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()
plt.tight_layout()
plt.show()