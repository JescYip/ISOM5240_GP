import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import os

# 1. 页面配置 (必须作为第一个 Streamlit 命令)
st.set_page_config(
    page_title="ISOM5240 Retail AI Assistant", 
    page_icon="🛍️",
    layout="centered"
)

# UI 标题与简介
st.title("🛍️ 智能零售营销助手 (微调版)")
st.write("当前运行：微调后的 ViT (识别) + GPT-2 (营销文案生成)。")

# 2. 加载模型 (指向你刚才保存的本地路径)
@st.cache_resource
def load_pipelines():
    # --- 修改点 1：指向微调后的本地文件夹 ---
    # 确保这两个文件夹与 app.py 在同一个目录下，或者使用绝对路径
    vit_path = "JescYip/vit-retail-finetuned"
    gpt2_path = "JescYip/gpt2-ad-finetuned"
    
    # 加载图像识别 Pipeline
    image_classifier = pipeline("image-classification", model=vit_path)
    
    # 加载文本生成 Pipeline
    text_generator = pipeline("text-generation", model=gpt2_path)
    
    return image_classifier, text_generator

# 如果本地文件夹不存在，提示错误（IT 咨询中的健壮性检查）
try:
    v_pipe, t_pipe = load_pipelines()
except Exception as e:
    st.error(f"模型加载失败！请确保本地存在微调文件夹。错误详情: {e}")
    st.stop()

# 3. 上传图片组件
uploaded_file = st.file_uploader("上传商品图片以开始营销自动化...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='待处理商品', use_container_width=True)
    
    st.divider()
    
    # --- 第一步：商品识别 (Keyword Identification) ---
    with st.spinner('微调模型正在精准识别...'):
        v_results = v_pipe(image)
        top_label = v_results[0]['label']
        confidence = v_results[0]['score']
        
    st.subheader("第一步：商品精准识别")
    st.success(f"识别类别: **{top_label}** (置信度: {confidence:.2%})")
    
    # --- 第二步：生成广告词 (修正版) ---
    with st.spinner('正在创作广告文案...'):
        # 1. 必须匹配训练时的格式，不要加多余的描述词
        # 注意 \n 前后的空格要跟训练代码完全一致
        prompt = f"Product: {top_label} \n Ad:"
        
        # 2. 调用模型
        t_results = t_pipe(
            prompt, 
            max_length=150,          # 稍微长一点，防止话没说完
            num_return_sequences=1, 
            truncation=True,
            do_sample=True,          # 开启随机采样，增加创意
            temperature=0.7,         # 控制随机性
            top_k=50,                # 限制候选词，提高质量
            pad_token_id=50256
        )
        
        # 3. 稳健的提取逻辑：直接切分字符串
        full_output = t_results[0]['generated_text']
        
        # 调试用：如果还是没输出，可以取消下面这行的注释看看模型到底输出了什么
        # st.write(f"DEBUG: {full_output}") 
    
        if " Ad:" in full_output:
            # 获取 "Ad:" 之后的所有内容
            generated_text = full_output.split(" Ad:")[-1].strip()
            # 去掉可能存在的结束符
            generated_text = generated_text.replace("<|endoftext|>", "")
        else:
            # 如果没找到分隔符，就尝试去掉 prompt 本身
            generated_text = full_output.replace(prompt, "").strip()
    
    st.subheader("第二步：广告生成 (Ad Generation)")
    if generated_text:
        st.info(generated_text)
    else:
        st.warning("模型未能生成有效文案，请尝试重新上传或检查模型微调情况。")

    # 展示逻辑解题方法 (Logical Approach)
    with st.expander("查看技术闭环 (Technical Business Logic)"):
        st.write("1. **CV Optimization**: 通过微调，ViT 已从通用识别转变为专注零售 141 类的专家模型。")
        st.write(f"2. **NLP Alignment**: GPT-2 已学习了针对 '{top_label}' 类别的营销话术。")
        st.write("3. **End-to-End Value**: 实现了从非结构化图片到结构化营销内容的自动化产出。")
