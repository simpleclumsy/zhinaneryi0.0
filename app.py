# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:27:20 2026

@author: 95326
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:32:44 2026

@author: 95326
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import os
# 指定字体路径（与 app.py 同级的 fonts/SimHei.ttf）
font_path = os.path.join(os.path.dirname(__file__), "fonts", "SimHei.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    # 设置全局字体
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示异常
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import re
from io import StringIO
from analyzer import TranslationAnalyzer  # 你的核心分析类
import spacy



# ++++++++++++++ 新增：指标解释字典 ++++++++++++++
METRIC_EXPLANATIONS = {
    '语义精确度': '名词义项数总和与名词总数之商。值越高，说明词语的集中程度越高，则阅读难度更低。',
    '词汇丰富度': '基于词汇频率的熵.值越大说明词汇使用越多样，文本越富于变化，阅读难度可能更高。',
    '句法丰富度': '基于依存关系种类的熵.值越大说明文本的依存关系或句法结构越不确定，文本的句法越富于变化，则文本越难读。',
    '语义丰富度': '基于名词分布的熵.值越大说明文本的话题越丰富，文本的可读性可能越低。',
    '语义清晰度': '名词概率分布的偏度。语义清晰度值越大，说明文本以名词为代表的话题越集中，其语义越清晰。',
    '语义噪音': '语义噪音值越大，说明文本以名词为代表的话题越偏向不重要的话题，其语义噪音越大。',
    '平均句长': '每个句子包含的平均词数。句长越长，句法可能越复杂，阅读负担越大。',
    '移动窗口TTR(MATTR)': '词汇多样性的移动窗口测量，值越高表示词语重复越少，词汇变化越丰富。',
    '虚词占比': '虚词（介词、连词、助词等）在总词数中的比例，与文本的抽象程度和衔接方式有关。',
    '平均依存距离': '依存语法中支配词与被支配词的平均线性距离，距离越大通常句子结构越复杂。'
}
# +++++++++++++++++++++++++++++++++++++++++++++

# ===================== 页面设置 =====================
st.set_page_config(
    page_title="知难而易",
    # page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# 通过 URL 参数切换页面
params = st.query_params
if "page" in params:
    current_page = params["page"]
else:
    current_page = "文本难易度检测"
# 居中艺术字标题
st.markdown(
    """
    <h1 style="
        text-align: center;
        font-size: 3.5em;
        font-family: 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', sans-serif;
        background: linear-gradient(135deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    ">知难而易</h1>
    """,
    unsafe_allow_html=True
)
# ===================== 自定义导航栏 =====================
# 动态 active 类
tab_active = "active" if current_page == "文本难易度检测" else ""
tab2_active = "active" if current_page == "尽情期待" else ""

st.markdown(
    f"""
    <style>
    .topnav {{
        background-color: #f8f9fa;
        padding: 10px 20px;
        border-bottom: 2px solid #dee2e6;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .topnav a {{
        text-decoration: none;
        color: #343a40;
        font-size: 18px;
        margin-right: 30px;
        padding: 5px 0;
        border-bottom: 2px solid transparent;
    }}
    .topnav a:hover, .topnav a.active {{
        border-bottom: 2px solid #1f77b4;
    }}
    </style>
    <div class="topnav">
        <a href="?page=文本难易度检测" class="{tab_active}">文本难易度检测</a>
        <a href="?page=尽情期待" class="{tab2_active}">尽情期待</a>
    </div>
    """,
    unsafe_allow_html=True
)

if current_page == "尽情期待":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: center; color: #666;'>🚧 更多精彩功能，敬请期待…</h2>",
        unsafe_allow_html=True
    )
    st.stop()
# 否则，继续显示“文本难易度检测”主界面（原有代码无需改动）


# ===================== 页面1：文本难易度检测 =====================
st.markdown("###  上传或输入待分析的翻译文本")

# ----- 全局语言选择 -----


# ----- 输入方式选择 -----
input_mode = st.radio(
    "选择输入方式",
    ["📁 上传文件（.txt）", "✍️ 手动输入"],
    horizontal=True
)

# 存储最终文本的字典
texts_dict = {}

# ===================== 方式1：上传文件 =====================
if input_mode == "📁 上传文件（.txt）":
    uploaded_files = st.file_uploader(
        "上传翻译文本文件（支持 .txt）",
        type=["txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.info("请为每个文件选择对应的语言：")
        # 初始化语言选择状态
        if "file_lang_dict" not in st.session_state:
            st.session_state.file_lang_dict = {}
        # 展示每个文件
        for f in uploaded_files:
            fname = f.name
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.text(f"📄 {fname}")
            with col2:
                cur_lang = st.session_state.file_lang_dict.get(fname, "zh")
                lang = st.selectbox(
                    f"语言 ({fname})",
                    ["中文", "英语", "日语"],
                    index=["中文", "英语", "日语"].index(
                        {"zh": "中文", "en": "英语", "ja": "日语"}.get(cur_lang, "中文")
                    ),
                    key=f"lang_{fname}"
                )
                # 更新状态
                lang_code = {"中文": "zh", "英语": "en", "日语": "ja"}[lang]
                st.session_state.file_lang_dict[fname] = lang_code
        # 构建多语言字典供分析使用
        multi_lang_texts = {"zh": {}, "en": {}, "ja": {}}
        for f in uploaded_files:
            lang_code = st.session_state.file_lang_dict.get(f.name, "zh")
            model_name = os.path.splitext(f.name)[0]
            content = f.read().decode("utf-8")
            # 存储该模型的文本列表
            if model_name not in multi_lang_texts[lang_code]:
                multi_lang_texts[lang_code][model_name] = []
            multi_lang_texts[lang_code][model_name].append(content)
        # 保存到 session_state 以便分析时使用
        st.session_state.multi_lang_texts = multi_lang_texts
        st.success(f"✅ 已上传 {len(uploaded_files)} 个文件，语言已设定。")
        
# ===================== 方式2：手动输入 =====================
else:
    # 初始化输入框数量
    if "num_inputs" not in st.session_state:
        st.session_state.num_inputs = 1

    # 添加文本框的函数
    def add_input():
        st.session_state.num_inputs += 1

    # 删除最后一个文本框
    def remove_input():
        if st.session_state.num_inputs > 1:
            # 清除被删文本框的 session state 值
            i = st.session_state.num_inputs - 1
            st.session_state.pop(f"model_name_{i}", None)
            st.session_state.pop(f"text_{i}", None)
            st.session_state.num_inputs -= 1

    # 手动输入区域
    st.markdown("**为每个模型输入一段文本（建议每个模型提供多段文本以上传文件方式实现）**")
    st.caption("目前每个文本框代表一个模型的一段文本，统计检验可能因样本量不足而无法进行。")

    # 文本框布局
    cols_per_row = st.columns([0.7, 0.15, 0.15])
    for i in range(st.session_state.num_inputs):
        # 模型名称输入
        model_key = f"model_name_{i}"
        text_key = f"text_{i}"
        default_model = f"模型{i+1}"
        model = st.text_input(
            f"模型名称 {i+1}",
            value=default_model,
            key=model_key,
            label_visibility="collapsed"
        )
        # 文本输入
        text = st.text_area(
            f"文本 {i+1}",
            placeholder="请输入待分析文本…",
            key=text_key,
            height=100,
            label_visibility="collapsed"
        )
            # 在 for i in range(st.session_state.num_inputs): 循环内，model 和 text 输入之后
        lang_key = f"lang_{i}"
        if lang_key not in st.session_state:
            st.session_state[lang_key] = "中文"
        lang_choice = st.selectbox(
            f"语言 {i+1}",
            ["中文", "英语", "日语"],
            index=["中文", "英语", "日语"].index(st.session_state[lang_key]),
            key=f"select_lang_{i}"
        )
        st.session_state[lang_key] = lang_choice
    # 控制按钮
    col1, col2 = st.columns([0.15, 0.15])
    with col1:
        st.button("➕ 添加文本", on_click=add_input, use_container_width=True)
    with col2:
        st.button("➖ 移除最后", on_click=remove_input, use_container_width=True)

    # 构建 texts_dict（在点击“开始分析”时使用）
    # 这里暂时先收集当前输入的值预览（可选）
    # if st.session_state.num_inputs >= 2:
    #     st.caption("当前已输入模型示例（点击“开始分析”后生效）")
    #     preview = {}
    #     for i in range(st.session_state.num_inputs):
    #         m = st.session_state.get(f"model_name_{i}", f"模型{i+1}")
    #         t = st.session_state.get(f"text_{i}", "").strip()
    #         if t:
    #             preview[m] = [t]
    #     if preview:
    #         st.json(preview)

# ===================== 开始分析按钮 =====================
st.markdown("---")
# 取消禁用，所有语言均可分析
analyze_btn = st.button("开始分析", type="primary", use_container_width=True)



    # 2. 构建 texts_dict（上传模式已构建，手动模式在此构建）
if analyze_btn:
    # 构建多语言字典
    multi_lang_texts = {"zh": {}, "en": {}, "ja": {}}
    if input_mode == "📁 上传文件（.txt）":
        # 使用之前存储的 st.session_state.multi_lang_texts
        if "multi_lang_texts" in st.session_state:
            multi_lang_texts = st.session_state.multi_lang_texts
        else:
            st.error("请先上传文件并设定语言。")
            st.stop()
    else:  # 手动输入
        for i in range(st.session_state.num_inputs):
            m = st.session_state.get(f"model_name_{i}", f"模型{i+1}").strip()
            t = st.session_state.get(f"text_{i}", "").strip()
            lang_choice = st.session_state.get(f"lang_{i}", "中文")
            lang_code = {"中文": "zh", "英语": "en", "日语": "ja"}[lang_choice]
            if t:
                if m not in multi_lang_texts[lang_code]:
                    multi_lang_texts[lang_code][m] = []
                multi_lang_texts[lang_code][m].append(t)

    # 确保至少有2个模型
    total_models = sum(len(v) for v in multi_lang_texts.values())
    if total_models < 2:
        st.error("❌ 至少需要两个模型的有效文本才能对比。")
        st.stop()

    with st.spinner("⏳ 正在分析，请稍候……"):
        analyzer = TranslationAnalyzer(multi_lang_texts)
        long_df = analyzer.compute_all_metrics()
        desc_long, stat_df = analyzer.generate_statistical_tables(long_df)

    # 存储结果...



    # 4. 存储到 session_state
    st.session_state.analysis_done = True
    st.session_state.long_df = long_df
    st.session_state.desc_long = desc_long
    st.session_state.stat_df = stat_df
    st.success("分析完成！")
# ===================== 结果展示区域 =====================
if st.session_state.get("analysis_done", False):
    desc_long = st.session_state.desc_long
    stat_df = st.session_state.stat_df

    st.markdown("---")
    st.header("分析结果")
    
   
        # ------------- 定义分组顺序（用于排序和展示） -------------
    grouping_order = [
        ('精确度', ['动词语义精确度', '名词语义精确度', '名词与动词语义精确度', '实词语义精确度']),
        ('丰富度', ['词汇丰富度', '句法丰富度', '语义丰富度']),
        ('语义清晰度', ['语义清晰度']),
        ('语义噪音', ['语义噪音']),
        ('平均句长', ['平均句长']),
        ('虚词占比', ['虚词占比']),
        ('移动窗口TTR', ['移动窗口TTR(MATTR)']),
        ('平均依存距离', ['平均依存距离']),   # ← 这里追加
    ]
    all_ordered_metrics = [m for _, metrics in grouping_order for m in metrics]
    # 当前数据中实际存在的指标（但限定为分组内的指标）
    available_metrics_in_data = [m for m in all_ordered_metrics if m in desc_long['指标'].unique()]

    # ------------- 侧边栏筛选控件 -------------
    with st.sidebar:
        st.header("⚙️ 筛选与展示")
        selected_metrics = st.multiselect(
            "选择指标",
            available_metrics_in_data,
            default=available_metrics_in_data[:min(6, len(available_metrics_in_data))]
        )
        show_stats = st.multiselect(
            "选择统计量",
            ['最小值', '最大值', '中位数', '平均值', '标准差', 'P值', '显著性'],
            default=['平均值', 'P值', '显著性']
        )
        sig_filter = st.radio(
            "显著性水平",
            ['全部', '仅显著 (p<0.05)', '显著+边缘显著 (p<0.1)'],
            index=0
        )

        # ------------- 应用筛选 -------------
    if selected_metrics:
        desc_sub = desc_long[desc_long['指标'].isin(selected_metrics)]
    else:
        desc_sub = desc_long

    if sig_filter == '仅显著 (p<0.05)':
        sig_list = stat_df[stat_df['P值'] < 0.05]['指标'].tolist()
        desc_sub = desc_sub[desc_sub['指标'].isin(sig_list)]
    elif sig_filter == '显著+边缘显著 (p<0.1)':
        sig_list = stat_df[stat_df['P值'] < 0.1]['指标'].tolist()
        desc_sub = desc_sub[desc_sub['指标'].isin(sig_list)]

    if desc_sub.empty:
        st.warning("当前筛选条件下无指标，请调整选择。")
        st.stop()

    # 添加分组列并排序（只保留分组内的指标）
    metric_to_group = {m: group for group, metrics in grouping_order for m in metrics}
    desc_sub = desc_sub[desc_sub['指标'].isin(all_ordered_metrics)]  # 去除无关指标
    desc_sub['分组'] = desc_sub['指标'].map(metric_to_group)
    # 按分组顺序和指标顺序排序
    desc_sub['指标'] = pd.Categorical(desc_sub['指标'], categories=all_ordered_metrics, ordered=True)
    desc_sub['分组'] = pd.Categorical(desc_sub['分组'], categories=[g[0] for g in grouping_order], ordered=True)
    desc_sub = desc_sub.sort_values(['分组', '指标'])

    # ------------- 构建带分组的多级索引宽表 -------------
    wide = desc_sub.pivot(index=['分组', '指标'], columns='模型',
                         values=['最小值', '最大值', '中位数', '平均值', '标准差'])
    stat_order = ['最小值', '最大值', '中位数', '平均值', '标准差']
    wide = wide.reindex(stat_order, level=0, axis=1)
    wide = wide.swaplevel(axis=1).sort_index(axis=1)

    # 扁平化列名（双下划线分隔）
    flat_wide = wide.copy()
    flat_wide.columns = ['__'.join(col).strip() for col in flat_wide.columns.values]

    # 添加 P 值和显著性（索引已是多级）
    p_dict = dict(zip(stat_df['指标'], stat_df['P值']))
    flat_wide['P值'] = [p_dict.get(idx[1], np.nan) for idx in flat_wide.index]   # idx[1] 是具体指标名
    def significance_label(p):
        if pd.isna(p):
            return 'N/A'
        if p < 0.05:
            return '显著差异'
        elif p < 0.10:
            return '边缘显著差异'
        else:
            return '无显著差异'
    flat_wide['显著性'] = [significance_label(p_dict.get(idx[1], np.nan)) for idx in flat_wide.index]

    # 仅保留用户选择的统计量
    retain_cols = []
    for col in flat_wide.columns:
        if col in ['P值', '显著性']:
            if col in show_stats:
                retain_cols.append(col)
        else:
            parts = col.rsplit('__', 1)
            stat_part = parts[-1] if len(parts) == 2 else ''
            if stat_part in show_stats:
                retain_cols.append(col)
    flat_wide = flat_wide[retain_cols]
    flat_wide = flat_wide.round(2)

    # ------------- 行高亮样式 -------------
    def highlight_row(row):
        if row['显著性'] == '显著差异':
            return ['background-color: #d4edda'] * len(row)
        elif row['显著性'] == '边缘显著差异':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return [''] * len(row)

    styled_df = flat_wide.style.apply(highlight_row, axis=1)
    
    # ++++++++++++++ 新增：折叠式指标说明面板 ++++++++++++++
    with st.expander("点击查看指标说明"):
        for metric, explanation in METRIC_EXPLANATIONS.items():
            st.markdown(f"**{metric}**：{explanation}")
# +++++++++++++++++++++++++++++++++++++++++++++++++
    st.subheader("综合统计表")
    st.dataframe(styled_df, use_container_width=True)

    # ------------- 准备图表所需数据 -------------
    # 使用 desc_sub 生成 pivot_mean 和 pivot_std
    pivot_mean = desc_sub.pivot(index='指标', columns='模型', values='平均值')
    pivot_std = desc_sub.pivot(index='指标', columns='模型', values='标准差')
    # 有效指标（仅用于图表）：至少有一个模型有有效平均值
    valid_key = [
        m for m in all_ordered_metrics
        if m in desc_sub['指标'].unique() and m in pivot_mean.index and pivot_mean.loc[m].notna().any()
    ]

    # ------------- 绘图函数（移植并返回 fig） -------------
    def plot_single_metric(metric_name, p_val, means, stds):
        valid_models = means.dropna().index
        means_clean = means[valid_models]
        stds_clean = stds[valid_models]

        # 数据不足时：生成占位图并返回
        if len(means_clean) == 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.text(0.5, 0.5, '数据不足，无法绘制',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            ax.set_title(f'{metric_name}\n(暂无有效数据)', fontsize=12)
            fig.tight_layout()
            return fig
            
        models = means_clean.index.tolist()
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(x, means_clean, width=0.6, yerr=stds_clean, capsize=5,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)],
                      tick_label=models, alpha=0.8, edgecolor='black')
        for bar, mv in zip(bars, means_clean):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{mv:.2f}', ha='center', va='bottom', fontsize=9)
        low_vals = means_clean - stds_clean
        high_vals = means_clean + stds_clean
        min_val = low_vals.min()
        max_val = high_vals.max()
        if max_val - min_val < 1e-8:
            min_val -= 0.1
            max_val += 0.1
        else:
            padding = 0.1*(max_val - min_val)
            min_val -= padding
            max_val += padding
        ax.set_ylim(min_val, max_val)
        ax.set_ylabel('指标值')
        p_str = f'{p_val:.3f}' if not pd.isna(p_val) else 'N/A'
        ax.set_title(f'{metric_name}\n(P值 = {p_str})', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        return fig

    def plot_heatmap(pivot_mean_norm, annot_df):
        fig, ax = plt.subplots(figsize=(max(6, len(pivot_mean_norm.columns)*1.2),
                                        max(4, len(pivot_mean_norm.index)*0.6)))
        sns.heatmap(pivot_mean_norm, annot=annot_df, fmt='', cmap='RdBu_r', center=0,
                    cbar_kws={'label': '标准化分数 (Z-score)'}, linewidths=0.5, ax=ax)
        ax.set_title('关键指标相对表现（Z-score标准化）', fontsize=14)
        ax.set_ylabel('指标')
        ax.set_xlabel('模型')
        fig.tight_layout()
        return fig

    def plot_placeholder(title, message="数据不足，无法绘制"):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, message, ha='center', va='center',
                fontsize=14, color='gray', transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig
        
    def plot_radar(radar_data, valid_keys, LOW=0.2, HIGH=1.0):
        radar_norm = radar_data.copy()
        for metric in radar_norm.index:
            row = radar_data.loc[metric]
            min_v, max_v = row.min(), row.max()
            if max_v - min_v < 1e-8:
                radar_norm.loc[metric] = (LOW + HIGH) / 2
            else:
                radar_norm.loc[metric] = LOW + (row - min_v) / (max_v - min_v) * (HIGH - LOW)
        angles = [n / float(len(valid_keys)) * 2 * pi for n in range(len(valid_keys))]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for model in radar_norm.columns:
            values = radar_norm[model].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(valid_keys, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_title(f'雷达图（Min-Max归一化至{LOW}-{HIGH}）', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        fig.tight_layout()
        return fig

    # ------------- 图表选项卡 -------------
    tab1, tab2, tab3 = st.tabs(["📈 柱状图", "🔥 热图", "🕸️ 雷达图"])

    with tab1:
        if len(valid_key) > 0:
            metric_to_plot = st.selectbox("选择要查看的指标", valid_key, key="bar_metric")
            p_val = p_dict.get(metric_to_plot, np.nan)
            means_series = pivot_mean.loc[metric_to_plot]
            stds_series = pivot_std.loc[metric_to_plot]
            fig = plot_single_metric(metric_to_plot, p_val, means_series, stds_series)
            if fig:
                st.pyplot(fig)
        else:
            st.info("没有可显示的指标。")

    with tab2:
    # 准备热图数据，剔除含有任何 NaN 的指标（确保标准化安全）
        pivot_heat = pivot_mean.loc[valid_key].dropna(how='any')
        if len(pivot_heat) < 2:
            fig = plot_placeholder("热图（指标不足）", "至少需要两个完整指标才能绘制热图")
            st.pyplot(fig)
        else:
            # Z-score标准化
            pivot_norm_key = pivot_heat.copy()
            for idx in pivot_norm_key.index:
                row = pivot_heat.loc[idx]
                mean_row = row.mean()
                std_row = row.std()
                if std_row == 0 or np.isnan(std_row):
                    pivot_norm_key.loc[idx] = 0.0
                else:
                    pivot_norm_key.loc[idx] = (row - mean_row) / std_row
            annot_vals = pivot_heat.round(2).astype(str)
            fig = plot_heatmap(pivot_norm_key, annot_vals)
            st.pyplot(fig)
            
    with tab3:
    # 准备雷达图数据，剔除含有任何 NaN 的指标（避免归一化失败）
        radar_clean = pivot_mean.loc[valid_key].dropna(how='any')
        if len(radar_clean) < 3:
            fig = plot_placeholder("雷达图（指标不足）", f"至少需要3个完整指标，当前仅{len(radar_clean)}个")
            st.pyplot(fig)
        else:
            fig = plot_radar(radar_clean, radar_clean.index.tolist())
            st.pyplot(fig)
