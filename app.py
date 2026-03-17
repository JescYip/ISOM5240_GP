import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# 设置页面标题
# 设置页面标题和图标
# 必须是独立的行
st.set_page_config(page_title="ISOM5240 Retail AI Assistant", page_icon="🛍️")

# 换行后写标题
st.title("🛍️ 智能零售营销助手")
st.write("上传一张商品图片，AI 将自动识别类别并生成广告词。")

# 1. 加载模型 (使用缓存避免重复加载)
@st.cache_resource
def load_pipelines():
    # Pipeline 1: 图像分类 (Image Classification)
    # 使用训练前的原始模型: google/vit-base-patch16-224
    # image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    image_classifier = pipeline("image-classification", model="SCM1120/blip-fashion-finetuned")
    
    # Pipeline 2: 文本生成 (Text Generation)
    # 使用训练前的原始模型: gpt2
    # text_generator = pipeline("text-generation", model="gpt2")
    text_generator = pipeline("text-generation", model="SCM1120/gpt2-ad-finetuned")

    return image_classifier, text_generator

v_pipe, t_pipe = load_pipelines()

# 2. 上传图片组件
uploaded_file = st.file_uploader("选择商品图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 展示图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的商品图片', use_container_width=True)
    
    st.divider()
    
    # --- 第一步：生成关键词简介 ---
    with st.spinner('正在识别商品类别...'):
        # 进行推理
        v_results = v_pipe(image)
        # 获取概率最高的 Top-1 结果
        top_label = v_results[0]['label']
        confidence = v_results[0]['score']
        
    st.subheader("第一步：商品识别 (Keyword Identification)")
    st.success(f"识别结果: **{top_label}** (置信度: {confidence:.2%})")
    
    # --- 第二步：生成广告词 ---
    with st.spinner('正在创作广告文案...'):
        # 构造 Prompt：引导模型生成广告逻辑
        # 因为是原始 GPT-2，我们需要更明确的 Prompt 引导
        prompt = f"The following is a creative advertisement for a professional retail product.\nProduct: {top_label}\nAd Copy:"
        
        # 生成文本
        t_results = t_pipe(
            prompt, 
            max_length=100, 
            num_return_sequences=1, 
            truncation=True,
            pad_token_id=50256 # GPT-2 的结束符 ID
        )
        
        # 提取生成的内容（去掉 Prompt 部分）
        generated_text = t_results[0]['generated_text'].replace(prompt, "").strip()

    st.subheader("第二步：广告生成 (Ad Generation)")
    st.info(generated_text if generated_text else "正在努力构思中...")

    # 项目建议：展示逻辑解题方法 (Logical Approach)
    with st.expander("查看技术逻辑 (Technical Logic)"):
        st.write(f"1. **Computer Vision**: 使用 ViT 模型提取图像特征并映射到 ImageNet 1000 类。")
        st.write(f"2. **NLP Bridge**: 将识别出的 '{top_label}' 作为上下文输入给 GPT-2。")
        st.write(f"3. **Generative AI**: 通过自回归预测 (Autoregressive) 生成后续营销文本。")
