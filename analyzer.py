# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:31:09 2026

@author: 95326
"""

# -*- coding: utf-8 -*-
"""
TranslationAnalyzer — 后端分析引擎
支持：中文 (zh) / 英语 (en) / 日语 (ja)
新增：自动切分长文本，保证每个模型至少2个样本
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TranslationAnalyzer:
    # 语言配置
    LANGUAGE_CONFIG = {
        'zh': {
            'model_name': 'zh_core_web_sm',
            'func_pos': {'ADP', 'CCONJ', 'SCONJ', 'PART', 'DET'},  # 中文虚词 UD 标签
            'install_cmd': 'python -m spacy download zh_core_web_sm'
        },
        'en': {
            'model_name': 'en_core_web_sm',
            'func_pos': {'ADP', 'CCONJ', 'SCONJ', 'PART', 'DET', 'AUX'},  # 英语虚词含 AUX
            'install_cmd': 'python -m spacy download en_core_web_sm'
        },
        'ja': {
            'model_name': 'ja_core_news_sm',
            'func_pos': {'ADP', 'CCONJ', 'SCONJ', 'PART', 'DET', 'AUX'},  # 日语虚词
            'install_cmd': 'python -m spacy download ja_core_news_sm'
        }
    }

    def __init__(self, multi_lang_texts):
        """
        multi_lang_texts: {
            'zh': {'模型A': [文本1, 文本2], ...},
            'en': {...},
            'ja': {...}
        }
        """
        self.multi_lang_texts = multi_lang_texts
        self.nlps = {}          # 缓存已加载的模型
        self.texts_dict = {}    # 最终用于计算的文本字典（模型名已带语言后缀）

    # 合并所有语言的文本，模型名后加语言标识
        for lang, texts_dict in self.multi_lang_texts.items():
            if lang not in self.LANGUAGE_CONFIG:
                raise ValueError(f"不支持的语言：{lang}")
            lang_name = {'zh': '中文', 'en': '英语', 'ja': '日语'}.get(lang, lang)
            for model, texts in texts_dict.items():
                new_model = f"{model} ({lang_name})"
                # 如果已经存在同名的（不太可能），追加数字后缀
                if new_model in self.texts_dict:
                    new_model = f"{model} ({lang_name})_2"
                self.texts_dict[new_model] = [t.strip() for t in texts if t.strip()]





    def _get_nlp(self, lang):
        if lang in self.nlps:
            return self.nlps[lang]
        config = self.LANGUAGE_CONFIG[lang]
        import spacy
        import subprocess
        try:
            nlp = spacy.load(config['model_name'])
            print(f"[✅] 成功加载 {config['model_name']} 模型")
        except OSError:
            print(f"[⏳] 正在下载 {config['model_name']} 模型...")
            subprocess.run(config['install_cmd'], shell=True, check=True)
            nlp = spacy.load(config['model_name'])
            print(f"[✅] 下载完成，已加载 {config['model_name']} 模型")
        self.nlps[lang] = nlp
        return nlp

    def _split_at_sentence_boundaries(self, text, nlp, n_splits=3):
        doc = nlp(text)                                     # 使用传入的 nlp
        sent_starts = [sent.start_char for sent in doc.sents if sent.text.strip()]
        if len(sent_starts) < n_splits:
            if len(sent_starts) == 0:
                return [text]
            # 句子不足 n_splits 时，按实际句子返回
            segments = []
            for i in range(len(sent_starts)):
                start = sent_starts[i]
                end = sent_starts[i+1] if i+1 < len(sent_starts) else len(text)
                seg = text[start:end].strip()
                if seg:
                    segments.append(seg)
            return segments
    
        total_len = len(text)
        split_points = [total_len * (i+1) // n_splits for i in range(n_splits - 1)]
        chosen = []
        for p in split_points:
            boundary = next((s for s in sent_starts if s >= p), None)
            if boundary is None or boundary == 0:
                boundary = sent_starts[-1]
            if boundary not in chosen:
                chosen.append(boundary)
    
        chosen.sort()
        segments = []
        prev = 0
        for b in chosen:
            seg = text[prev:b].strip()
            if seg:
                segments.append(seg)
            prev = b
        last_seg = text[prev:].strip()
        if last_seg:
            segments.append(last_seg)
        return segments if len(segments) >= 2 else [text]

    def _segment_and_tag(self, text):
        doc = self.nlp(text)
        valid_tokens = [t for t in doc if not t.is_punct and not t.is_space]
        words = [t.text for t in valid_tokens]
        pos_list = [(t.text, t.pos_) for t in valid_tokens]
        dep_labels = [t.dep_ for t in valid_tokens if t.dep_ != 'punct']
        return words, pos_list, dep_labels, doc, valid_tokens

    def _entropy(self, counts):
        total = sum(counts.values())
        if total == 0: return 0.0
        probs = np.array(list(counts.values())) / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    # ================= 原有ARC指标 =================
    def _calc_lexical_richness(self, words): return self._entropy(Counter(words))
    def _calc_syntactic_richness(self, deps): return self._entropy(Counter(deps))
    def _calc_semantic_accuracy(self, pos_list, pos_types):
        filtered = [w for w, tag in pos_list if tag in pos_types]
        if not filtered: return 0.0
        freq = Counter(filtered)
        total_tokens = sum(freq.values())
        sense_sum = sum((1.5 + 0.8 * np.log2(c + 1)) * c for w, c in freq.items())
        return sense_sum / total_tokens

    def _calc_semantic_distribution(self, pos_list, target_pos):
        nouns = [w for w, tag in pos_list if tag in target_pos]
        if len(nouns) < 2:
            return {'richness': 0.0, 'clarity': 0.0, 'noise': 0.0}
        freq = Counter(nouns)
        total = len(nouns)
        probs = np.array(list(freq.values()), dtype=float) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return {
            'richness': entropy,
            'clarity': float(st.skew(probs)),
            'noise': float(st.kurtosis(probs))
        }

    # ================= 新增基础语言学指标 =================
    def _calc_mattr(self, tokens, window_size=50):
        if not tokens:
            return 0.0
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens)
        ttr_values = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            ttr_values.append(len(set(window)) / window_size)
        return float(np.mean(ttr_values))

    def _calc_basic_linguistic_metrics(self, doc, valid_tokens, lang_code):
        sents = [s for s in doc.sents if len([t for t in s if not t.is_punct]) > 0]
        if not valid_tokens or not sents:
            return {'平均句长': 0.0, '移动窗口TTR(MATTR)': 0.0, '虚词占比': 0.0, '平均依存距离': 0.0}
    
        words = [t.text for t in valid_tokens]
        total_tokens = len(valid_tokens)
    
        mattr = self._calc_mattr(words, window_size=50)
        avg_sent_len = total_tokens / len(sents)
    
        # 根据语言获取虚词标签集
        func_pos = self.LANGUAGE_CONFIG[lang_code]['func_pos']
        func_count = sum(1 for t in valid_tokens if t.pos_ in func_pos)
        func_ratio = func_count / total_tokens
    
        dep_dists = [abs(t.head.i - t.i) for t in valid_tokens if t.head != t]
        mean_dep_dist = float(np.mean(dep_dists)) if dep_dists else 0.0
    
        return {
            '平均句长': avg_sent_len,
            '移动窗口TTR(MATTR)': mattr,
            '虚词占比': func_ratio,
            '平均依存距离': mean_dep_dist
        }

    def compute_all_metrics(self):
        all_rows = []
        lang_name_to_code = {'中文': 'zh', '英语': 'en', '日语': 'ja'}
    
        for model, texts in self.texts_dict.items():
            # 从模型名解析语言代码
            lang_code = 'zh'
            for name, code in lang_name_to_code.items():
                if f"({name})" in model:
                    lang_code = code
                    break
    
            nlp = self._get_nlp(lang_code)
    
            for txt in texts:
                # 切分文本
                samples = self._split_at_sentence_boundaries(txt, nlp, n_splits=3)
                for sample in samples:
                    doc = nlp(sample)   # 注意：这里对切分后的每段重新解析，开销略大但安全
                    valid_tokens = [t for t in doc if not t.is_punct and not t.is_space]
                    words = [t.text for t in valid_tokens]
                    pos_list = [(t.text, t.pos_) for t in valid_tokens]
                    dep_labels = [t.dep_ for t in valid_tokens if t.dep_ != 'punct']
    
                    metrics = {
                        '词汇丰富度': self._calc_lexical_richness(words),
                        '句法丰富度': self._calc_syntactic_richness(dep_labels),
                        '名词语义精确度': self._calc_semantic_accuracy(pos_list, ['NOUN', 'PROPN']),
                        '动词语义精确度': self._calc_semantic_accuracy(pos_list, ['VERB']),
                        '名词与动词语义精确度': self._calc_semantic_accuracy(pos_list, ['NOUN', 'PROPN', 'VERB']),
                        '实词语义精确度': self._calc_semantic_accuracy(pos_list, ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']),
                    }
                    dist = self._calc_semantic_distribution(pos_list, ['NOUN', 'PROPN'])
                    metrics.update({
                        '语义丰富度': dist['richness'],
                        '语义清晰度': dist['clarity'],
                        '语义噪音': dist['noise']
                    })
                    basic = self._calc_basic_linguistic_metrics(doc, valid_tokens, lang_code)
                    metrics.update(basic)
    
                    for m, v in metrics.items():
                        all_rows.append({'模型': model, '指标': m, '值': v})
        return pd.DataFrame(all_rows)

    def generate_statistical_tables(self, long_df, output_dir='translation_analysis_output', key_metrics=None):
        os.makedirs(output_dir, exist_ok=True)
        models = long_df['模型'].unique()
        k = len(models)

        def significance_label(p):
            if pd.isna(p):
                return 'N/A'
            if p < 0.05:
                return '显著差异'
            elif p < 0.10:
                return '边缘显著差异'
            else:
                return '无显著差异'

        desc_long = long_df.groupby(['模型', '指标'])['值'].agg(['min', 'max', 'median', 'mean', 'std']).reset_index()
        desc_long.columns = ['模型', '指标', '最小值', '最大值', '中位数', '平均值', '标准差']
        desc_long = desc_long.round(2)

        if key_metrics is not None and len(key_metrics) > 0:
            desc_long = desc_long[desc_long['指标'].isin(key_metrics)]

        wide = desc_long.pivot(index='指标', columns='模型', values=['最小值', '最大值', '中位数', '平均值', '标准差'])
        stat_order = ['最小值', '最大值', '中位数', '平均值', '标准差']
        wide = wide.reindex(stat_order, level=0, axis=1)
        wide = wide.swaplevel(axis=1).sort_index(axis=1)
        wide = wide.round(2)

        ttest_rows = []
        for m in desc_long['指标'].unique():
            groups = [long_df[(long_df['模型'] == mod) & (long_df['指标'] == m)]['值'].values for mod in models]
            valid_groups = [g for g in groups if len(g) >= 2]
            if len(valid_groups) == k and k >= 3:
                f_stat, p_val = st.f_oneway(*valid_groups)
                df = sum(len(g) for g in valid_groups) - k
                ttest_rows.append({'指标': m, 'F值(统计量)': round(f_stat, 2), '自由度': int(df), 'P值': round(p_val, 2)})
            elif k == 2:
                t_res = st.ttest_ind(*valid_groups, equal_var=False)
                ttest_rows.append({'指标': m, 'T值(统计量)': round(t_res.statistic, 2), '自由度': int(t_res.df), 'P值': round(t_res.pvalue, 2)})
            else:
                ttest_rows.append({'指标': m, 'F值(统计量)': np.nan, '自由度': np.nan, 'P值': np.nan})

        stat_df = pd.DataFrame(ttest_rows)
        stat_df.rename(columns={'F值(统计量)': 'F值', 'T值(统计量)': 'T值'}, inplace=True)
        if 'F值' in stat_df.columns:
            stat_df = stat_df[['指标', 'F值', '自由度', 'P值']]
        else:
            stat_df = stat_df[['指标', 'T值', '自由度', 'P值']]

        p_dict = dict(zip(stat_df['指标'], stat_df['P值']))
        flat_wide = wide.copy()
        flat_wide.columns = ['_'.join(col).strip() for col in flat_wide.columns.values]
        flat_wide['P值'] = [p_dict.get(idx, np.nan) for idx in flat_wide.index]
        flat_wide['显著性'] = [significance_label(p_dict.get(idx, np.nan)) for idx in flat_wide.index]
        flat_wide = flat_wide.round({col: 2 for col in flat_wide.columns if col != 'P值'})
        flat_wide['P值'] = flat_wide['P值'].round(3)

        # 控制台打印（可在网页中关闭）
        print("\n" + "=" * 70)
        print(" 表1 关键指标描述性统计（宽格式）")
        print(wide.to_string())
        print("\n" + "=" * 70)
        print(f" 表2 {'方差分析(F检验)' if k >= 3 else '独立样本T检验'}结果")
        print(stat_df.to_string(index=False))
        print("\n" + "=" * 70)
        print(" 表3 关键指标描述统计与显著性检验（综合表）")
        print(flat_wide.to_string())
        print("=" * 70)

        # md_path = os.path.join(output_dir, 'Translation_Statistical_Report.md')
        # with open(md_path, 'w', encoding='utf-8') as f:
        #     f.write("### 表1 关键指标描述性统计（宽格式）\n\n")
        #     f.write(wide.to_markdown() + "\n\n")
        #     f.write(f"### 表2 {'方差分析(F检验)' if k >= 3 else 'T检验'}结果\n\n")
        #     f.write(stat_df.to_markdown(index=False) + "\n\n")
        #     f.write("### 表3 关键指标描述统计与显著性检验（综合表）\n\n")
        #     f.write(flat_wide.to_markdown() + "\n\n")
        # print(f"✅ 统计表格已保存至: {md_path}")

        return desc_long, stat_df


if __name__ == '__main__':
    # 本地测试（需替换为实际文件）
    print("📥 初始化分析器...")
    texts_dict = {
        'modelA': ["This is a test. This is only a test."],
        'modelB': ["Another example text. More text here."]
    }
    try:
        analyzer = TranslationAnalyzer(texts_dict, language='en')
        long_df = analyzer.compute_all_metrics()
        desc, stat = analyzer.generate_statistical_tables(long_df)
        print("✅ 测试通过")
    except Exception as e:
        print(f"❌ 错误：{e}")
