import os
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 下载NLTK必需语料库
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------- 文本预处理函数 --------------------------
# 函数功能：对英文评论文本进行预处理（分词、词形还原、停用词过滤），适配LDA主题挖掘
# 输入：texts - 待预处理的评论文本列表（list[str]）
# 输出：processed_texts - 预处理后的文本列表（list[list[str]]，每个元素为单条评论的分词结果）
def preprocess_text(texts):
    # 步骤1：加载停用词
    stop_words = set(stopwords.words('english'))
    # 自定义停用词：
    custom_stopwords = {'book', 'review', 'story', 'read', 'novel', 'one', 'would', 'even', 'get', 'go'}
    stop_words = stop_words.union(custom_stopwords)
    # 步骤2：初始化词形还原器
    lemmatizer = WordNetLemmatizer()
    processed_texts = []
    # 步骤3：逐文本预处理
    for text in texts:
        tokens = simple_preprocess(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in lemmatized_tokens]
        filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words and len(token) >= 3]
        processed_texts.append(filtered_tokens)
    return processed_texts

# -------------------------- LDA模型训练函数 --------------------------
# 函数功能：训练LDA主题模型，输出主题数量、困惑度、各主题核心词汇
# 输入：
#   data - 预处理后的文本列表（list[list[str]]，来自preprocess_text函数输出）
#   num_topics - 预设主题数量（int）
# 输出：
#   lda_model - 训练好的LDA模型（gensim.models.LdaModel）
#   corpus - 文本的词袋表示（list[list[tuple]]，LDA模型输入格式）
#   dictionary - 词汇-索引映射字典（gensim.corpora.Dictionary）
def train_lda_model(data, num_topics):
    try:
        # 步骤1：构建词汇字典（映射每个唯一词汇到整数索引）
        dictionary = corpora.Dictionary(data)
        # 步骤2：过滤极端词汇（减少噪声）
        dictionary.filter_extremes(no_below=5, no_above=0.8)
        # 步骤3：将文本转为词袋（BOW）格式
        corpus = [dictionary.doc2bow(text) for text in data]
        # 步骤4：训练LDA模型
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            update_every=1,
            chunksize=500,
            passes=50,
            alpha=0.7,
            eta=0.07,
            iterations=500,
            per_word_topics=True
        )
        # 步骤5：计算困惑度
        perplexity = lda_model.log_perplexity(corpus)
        print(f"主题数量: {num_topics}, 困惑度: {perplexity}")
        # 步骤6：输出各主题的核心词汇
        for topic_id in range(num_topics):
            words = lda_model.show_topic(topic_id, topn=10)
            print(f"  主题 {topic_id + 1}: {', '.join([word[0] for word in words])}")
        return lda_model, corpus, dictionary
    except Exception as e:
        print(f"训练模型时发生错误: {e}")
        return None, None, None

# -------------------------- LDA模型可视化函数 --------------------------
# 函数功能：生成LDA模型可视化结果（交互式HTML），直观展示主题分布
# 输入：
#   lda_model - 训练好的LDA模型（gensim.models.LdaModel）
#   corpus - 文本的词袋表示（list[list[tuple]]）
#   dictionary - 词汇-索引映射字典（gensim.corpora.Dictionary）
#   output_path - 可视化HTML文件保存路径（str）
# 输出：无（直接保存HTML文件）
def visualize_lda_model(lda_model, corpus, dictionary, output_path):
    try:
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, output_path)
        print(f"可视化结果已保存到: {output_path}")
    except Exception as e:
        print(f"生成可视化时发生错误: {e}")

# -------------------------- 主程序（模型训练+可视化） --------------------------
if __name__ == "__main__":
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    # ==================== 路径配置（使用者需将数据集放在代码同级目录） ====================
    file_path = os.path.join(desktop_path, "示例数据.xlsx")
    try:
        # 步骤1：读取数据集
        df = pd.read_excel(file_path)
        comments = df.iloc[:, 0].tolist()
        # 步骤2：文本预处理
        processed_comments = preprocess_text(comments)
        # 步骤3：遍历1-8个主题，训练模型并可视化
        for num_topics in range(1, 9):  # 遍历1-8个主题
            print(f"主题数量: {num_topics}")
            lda_model, corpus, dictionary = train_lda_model(processed_comments, num_topics=num_topics)
            if lda_model and corpus and dictionary:
                output_html_path = os.path.join(desktop_path, f"lda_visualization_{num_topics}_topics.html")
                visualize_lda_model(lda_model, corpus, dictionary, output_html_path)
    except FileNotFoundError:
        print("未找到指定的Excel文件，请检查文件路径和文件名。")
    except Exception as e:
        print(f"发生错误: {e}")