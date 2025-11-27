import pandas as pd
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 下载NLTK必需的语料库
nltk.download('punkt')
nltk.download('vader_lexicon')


# -------------------------- 词典读取函数 --------------------------
# 函数功能：读取AFINN情感词典
# 输入：file_path - AFINN词典文件路径（str）
# 输出：afinn_dict - 键为单词，值为情感得分的字典（dict）
def read_afinn_dict(file_path):
    afinn_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, score = parts
                    afinn_dict[word] = float(score)
    except FileNotFoundError:
        print(f"错误: AFINN词典文件 {file_path} 未找到。")
    return afinn_dict


# 函数功能：读取NRC情感词典
# 输入：file_path - NRC词典文件路径（str）
# 输出：nrc_dict - 键为单词，值为{情感标签: 0/1}的字典（dict）
def read_nrc_dict(file_path):
    nrc_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 10:
                    word = parts[0]
                    emotions = list(map(int, parts[1:]))
                    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                                      'negative', 'positive', 'sadness', 'surprise', 'trust']
                    nrc_dict[word] = dict(zip(emotion_labels, emotions))
    except FileNotFoundError:
        print(f"错误: NRC词典文件 {file_path} 未找到。")
    return nrc_dict


# 函数功能：读取自定义《西游记》领域情感词典
# 输入：file_path - 自定义词典文件路径（str）
# 输出：custom_dict - 键为单词，值为情感得分的字典（dict）
def read_custom_dict(file_path):
    custom_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, score = parts
                    custom_dict[word] = float(score)
    except FileNotFoundError:
        print(f"错误: 自定义词典文件 {file_path} 未找到。")
    return custom_dict



# -------------------------- 情感分析核心函数 --------------------------
# 函数功能：融合多词典的情感分析（自定义词典优先，AFINN/NRC/VADER加权融合）
# 输入：
#   text - 待分析的英文评论文本（str）
#   afinn_dict - AFINN词典字典（dict）
#   nrc_dict - NRC词典字典（dict）
#   custom_dict - 自定义领域词典字典（dict）
# 输出：
#   normalized_score - 归一化后的情感得分（float，范围[-1,1]）
def analyze_sentiment(text, afinn_dict, nrc_dict, custom_dict):
    # 步骤1：文本预处理
    tokens = word_tokenize(text.lower())
    total_score = 0.0
    vader_analyzer = SentimentIntensityAnalyzer()  # 初始化VADER分析器

    # 否定词/转折词权重参数
    negation_flag = 1  # 否定词标记：1=无否定，-1=有否定
    negation_window = 0  # 否定词影响窗口：否定词后2个词受影响
    transition_weight = 1.0  # 转折词权重：初始为1
    decay_rate = 0.1   # 转折词权重衰减率：每过一个词权重减0.1（最低1.0）

    # 步骤2：逐词计算情感得分（融合词典+否定/转折词调整）
    for token in tokens:
        # 子步骤1：处理否定词（not/never/no等）- 反转后续2个词的情感极性
        if token in ['not', 'never', 'no', 'don', 'dont']:
            negation_flag = -1
            negation_window = 2
        elif negation_window > 0:
            negation_window -= 1
        else:
            negation_flag = 1

        # 子步骤2：处理转折词（but/however等）
        if token in ['but', 'however', 'yet', 'though']:
            transition_weight = 1.5
        else:
            transition_weight = max(1.0, transition_weight - decay_rate)

        # 子步骤3：多词典融合计算单词语感得分（自定义词典优先）
        if token in custom_dict:
            # 优先级1：自定义《西游记》领域词典
            word_score = custom_dict[token]
        else:
            # 优先级2：通用词典加权融合
            afinn = afinn_dict.get(token, 0.0)
            nrc_entry = nrc_dict.get(token, {})
            nrc = nrc_entry.get('positive', 0) - nrc_entry.get('negative', 0)
            vader = vader_analyzer.polarity_scores(token)['compound']
            word_score = 0.4 * afinn + 0.4 * nrc + 0.2 * vader

        # 子步骤4：调整得分
        adjusted_score = word_score * negation_flag * transition_weight
        total_score += adjusted_score

    # 步骤3：归一化处理
    return math.tanh(total_score)



# -------------------------- 主程序（数据处理） --------------------------
if __name__ == "__main__":
    # ==================== 路径配置（需使用者根据本地路径修改） ====================
    # 词典文件路径（建议使用者将词典文件放在代码同级目录，直接填文件名）
    afinn_path = "AFINN-111.txt"
    nrc_path = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    custom_path = "custom_lexicon.txt"

    # 读取词典
    afinn_dict = read_afinn_dict(afinn_path)
    nrc_dict = read_nrc_dict(nrc_path)
    custom_dict = read_custom_dict(custom_path)

    # 数据集文件路径（建议使用者将数据集放在代码同级目录）
    data_path = "示例数据.xlsx"

    # 读取数据集并逐行分析情感
    try:
        df = pd.read_excel(data_path)
        scores = []
        sentiments = []

        for _, row in df.iterrows():
            text = str(row['processed_text'])
            raw_score = analyze_sentiment(text, afinn_dict, nrc_dict, custom_dict)

            # 情感倾向判定
            if raw_score > 0.1:  # 保持原缓冲阈值
                sentiment = '积极'
            elif raw_score < -0.1:
                sentiment = '消极'
            else:
                sentiment = '中性'

            scores.append(raw_score)
            sentiments.append(sentiment)

        # 保存结果
        df['融合情感得分'] = scores
        df['情感倾向'] = sentiments

        output_path = "示例数据_analyzed.xlsx"
        df.to_excel(output_path, index=False)
        print(f"分析结果已保存到 {output_path}")

    except Exception as e:
        print(f"处理时发生错误: {str(e)}")