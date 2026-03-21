import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# --- 页面配置 ---
st.set_page_config(page_title="ISOM5240 Retail AI Assistant", page_icon="🛍️", layout="wide")

st.title("🛍️ 智能零售营销助手 (Pro 版)")
st.write("集成 Swin-Transformer, BLIP 与 GPT-2 的多模态自动营销系统。")

# --- 1. 加载模型 (Pipeline 集成) ---
@st.cache_resource
def load_pipelines():
    # 1. 图像分类 (Swin-Tiny)
    classifier = pipeline("image-classification", model="JescYip/Swin-Tiny")
    
    # 2. 图像描述 (BLIP)
    captioner = pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")
    
    # 3. 广告生成 (GPT-2)
    ad_generator = pipeline("text-generation", model="SCM1120/gpt2-ad-finetuned")

    return classifier, captioner, ad_generator
    
with st.spinner('AI 引擎启动中...'):
    v_classifier, v_captioner, t_generator = load_pipelines()

# --- 2. 侧边栏与上传组件 ---
with st.sidebar:
    st.header("上传中心")
    uploaded_file = st.file_uploader("选择商品图片...", type=["jpg", "jpeg", "png"])
    st.info("建议：使用背景干净的电商产品图效果最佳。")

# --- 3. 主交互逻辑 ---
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='待识别商品', use_container_width=True)

    with col2:
        st.subheader("第一步：深度特征提取")
        
        # --- A. 运行 Swin-Tiny 分类 ---
        with st.spinner('Swin-Tiny 正在分析类别...'):
            cls_results = v_classifier(image)
            top_label = cls_results[0]['label']
            cls_confidence = cls_results[0]['score']
        
        # --- B. 运行 BLIP 生成描述 ---
        with st.spinner('BLIP 正在生成视觉描述...'):
            cap_results = v_captioner(image, text="")
            # 获取完整描述用于广告生成
            full_description = cap_results[0]['generated_text']
            keywords = ", ".join(full_description.split()[:10]) # 提取前10个词以提供更多上下文

        # 展示第一步结果
        st.success(f"**商品类别**: {top_label}")
        st.write(f"**视觉描述**: `{full_description}`")
        st.caption(f"分类置信度: {cls_confidence:.2%}")

        st.divider()

        # --- 第二步：GPT-2 广告生成 ---
        st.subheader("第二步：智能文案创作")
        with st.spinner('GPT-2 正在构思广告语...'):
            # 改进Prompt：使用更自然的语言，避免模板化
            prompt = f"Imagine you're writing a catchy slogan for a {top_label} with these features: {full_description}. Create an exciting and persuasive ad copy that highlights the benefits and makes people want to buy it:"
            
            ad_results = t_generator(
                prompt,
                max_length=150,
                min_length=50,
                num_return_sequences=3,
                truncation=True,
                temperature=0.8,
                pad_token_id=50256,
                do_sample=True,
                no_repeat_ngram_size=2
            )
            
            # 后处理：移除可能的模板元素
            ad_text = ad_results[0]['generated_text'].replace(prompt, "").strip()
            # 移除常见的广告模板前缀和后缀
            ad_text = ad_text.replace("Ad:", "").replace("#", "").strip()
            # 如果还有其他不想要的部分，可以继续清理

        st.info(ad_text if ad_text else "正在构思中...")

    # --- 4. 技术架构说明 (符合 ISOM5240 项目要求) ---
    with st.expander("查看项目逻辑架构 (Technical Pipeline Logic)"):
        st.markdown(f"""
        1.  **Swin-Tiny (Vision)**: 采用层次化 Transformer 架构对商品进行精确 3 分类（上装/下装/鞋子）。
        2.  **BLIP (Visual-Language)**: 负责 Bridge（桥接），将图像特征转化为非结构化的视觉描述词汇。
        3.  **GPT-2 (Generative AI)**: 接收 `Category + Keywords` 的多维输入，通过自回归生成符合电商逻辑的营销文案。
        """)
