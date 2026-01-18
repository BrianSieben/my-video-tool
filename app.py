import streamlit as st
import cv2
import os
import shutil
import numpy as np
from scenedetect import detect, ContentDetector
import google.generativeai as genai
import json # 用于格式化输出 JSON

# --- 1. 页面配置与高级美化 ---
st.set_page_config(page_title="VisionShot AI Pro", layout="wide", page_icon="🎬")

# 注入自定义 CSS，增强视觉质感
st.markdown("""
    <style>
    .stApp { background-color: #f8fafd; font-family: 'Inter', -apple-system, sans-serif; }
    .main-title {
        font-size: 3rem !important; font-weight: 800;
        background: -webkit-linear-gradient(#1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    /* 镜头容器样式 */
    .shot-header {
        background-color: #1e3a8a; color: white; padding: 5px 15px;
        border-radius: 5px; margin-top: 20px; margin-bottom: 10px;
    }
    /* 分析按钮的样式 */
    .stButton>button[key*="analyze_btn"] {
        background-color: #6c757d; /* 灰色 */
        border-color: #6c757d;
        color: white;
        margin-top: 5px;
        padding: 5px 10px;
        font-size: 0.8rem;
    }
    .stButton>button[key*="analyze_btn"]:hover {
        background-color: #5a6268;
        border-color: #5a6268;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Gemini API 配置 ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-pro-vision')
except Exception as e:
    st.error(f"⚠️ Gemini API 配置错误，请检查您的 Streamlit Secrets。详细错误：{e}")
    GEMINI_MODEL = None

# --- 3. Gemini 图片分析函数 ---
def analyze_image_with_gemini(image_data):
    if GEMINI_MODEL is None:
        return {"error": "Gemini 模型未初始化，无法进行图片分析。"}

    prompt_parts = [
        "你是一个专业的图像分析师和市场营销专家。请详细分析以下图片，输出一个 JSON 格式的结构化数据，用于描述图片的视觉风格、内容、光照、纹理、氛围，并生成两个潜在的英文prompt。",
        "图片分析维度（仅参考，可自行判断增减）：",
        "{",
        "  \"visual_style_analysis\": {",
        "    \"overall_mood\": \"\",",
        "    \"color_palette\": {",
        "      \"dominant_colors\": [],",
        "      \"temperature\": \"\",",
        "      \"saturation\": \"\"",
        "    },",
        "    \"composition\": {",
        "      \"framing\": \"\",",
        "      \"perspective\": \"\",",
        "      \"rule_of_thirds\": \"\",",
        "      \"action\": \"\"",
        "    }",
        "  },",
        "  \"content_analysis\": {",
        "    \"subjects\": \"\",",
        "    \"objects\": \"\",",
        "    \"scene\": \"\",",
        "    \"action\": \"\"",
        "  },",
        "  \"lighting_and_color\": {",
        "    \"lighting\": \"\",",
        "    \"color_contrast\": \"\"",
        "  },",
        "  \"texture_and_materials\": {",
        "    \"description\": \"\"",
    "  },",
        "  \"atmosphere_and_mood\": \"\",",
        "  \"potential_prompts\": []",
        "}",
        "请直接输出符合上述JSON结构的分析结果，不要包含任何额外文字或解释。"
    ]

    try:
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_data
        }
        response = GEMINI_MODEL.generate_content(prompt_parts + [image_part])
        # 尝试解析 Gemini 返回的文本为 JSON
        json_output = json.loads(response.text.strip())
        return json_output
    except json.JSONDecodeError as e:
        st.error(f"Gemini 返回的不是有效的 JSON 格式，请稍后再试或检查提示词。详细错误: {e}")
        st.code(response.text) # 显示原始响应以便调试
        return {"error": f"Gemini 返回格式错误：{e}"}
    except Exception as e:
        st.error(f"Gemini API 调用失败: {e}")
        return {"error": f"API调用失败: {e}"}

# --- 4. 侧边栏 ---
with st.sidebar:
    st.markdown("### 🛠️ 核心参数")
    threshold = st.slider("识别灵敏度", 10.0, 50.0, 27.0, help="数值越小，识别出的镜头越多")
    st.markdown("---")
    st.markdown("#### 功能说明")
    st.write("1. 自动识别镜头切换\n2. 每个镜头提取 4 帧\n3. 支持打包下载 ZIP\n4. **新增：AI 智能图片分析 (Powered by Gemini)**")
    st.caption("VisionShot AI v1.2")

# --- 5. 主界面布局 ---
st.markdown('<p class="main-title">🎬 VisionShot AI Pro</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>智能视频镜头拆解专家 - 现已支持每个镜头提取 4 帧预览及 AI 分析</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("请上传视频文件", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始深度分析", key="main_analysis_btn"):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        with st.spinner("AI 正在扫描并计算关键帧..."):
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            cap = cv2.VideoCapture(video_path)
            
            if not scene_list:
                st.warning("未能识别出明显的镜头切换，请调低灵敏度后再试。")
            else:
                for i, scene in enumerate(scene_list):
                    start_frame = scene[0].get_frames()
                    end_frame = scene[1].get_frames() - 1
                    duration = end_frame - start_frame
                    
                    mid_1 = start_frame + int(duration * 0.33)
                    mid_2 = start_frame + int(duration * 0.66)
                    
                    st.markdown(f'<div class="shot-header">🎞️ 镜头 {i+1:02d}</div>', unsafe_allow_html=True)
                    
                    cols = st.columns(4)
                    
                    extract_plan = [
                        (start_frame, '首帧'),
                        (mid_1, '过程1'),
                        (mid_2, '过程2'),
                        (end_frame, '尾帧')
                    ]
                    
                    for idx, (f_idx, label) in enumerate(extract_plan):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, frame = cap.read()
                        if ret:
                            img_name = f"shot_{i+1:03d}_{idx}_{label}.jpg"
                            img_path = os.path.join(output_dir, img_name)
                            cv2.imwrite(img_path, frame)
                            
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with cols[idx]:
                                st.image(frame_rgb, caption=f"{label} (F:{f_idx})", use_container_width=True)
                                
                                # --- 新增：图片分析按钮 ---
                                # 读取图片字节数据用于 Gemini
                                with open(img_path, "rb") as f:
                                    image_bytes_for_gemini = f.read()

                                if st.button("🖼️ 分析图片", key=f"analyze_btn_{i}_{idx}"):
                                    with st.spinner(f"AI 正在分析 {label} (F:{f_idx})..."):
                                        analysis_result = analyze_image_with_gemini(image_bytes_for_gemini)
                                        
                                        with st.expander(f"AI 分析结果 - 镜头 {i+1:02d} {label} (F:{f_idx})"):
                                            if "error" in analysis_result:
                                                st.error(analysis_result["error"])
                                            else:
                                                st.json(analysis_result)
                
                cap.release()
                st.success(f"✅ 处理完成！共分析出 {len(scene_list)} 个镜头。")
                st.balloons()

                shutil.make_archive("result_frames", 'zip', output_dir)
                with open("result_frames.zip", "rb") as f:
                    st.download_button(
                        label="📥 一键下载所有 4 帧截图 (ZIP)",
                        data=f,
                        file_name="visionshot_full_package.zip",
                        mime="application/zip"
                    )
else:
    st.info("👋 欢迎！上传视频后点击按钮，AI 会自动为您分析每一秒的内容。")
