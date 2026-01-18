import streamlit as st
import cv2
import os
import shutil
import numpy as np
from scenedetect import detect, ContentDetector

# --- 1. 页面配置 ---
st.set_page_config(page_title="VisionShot AI", layout="wide", page_icon="🎬")

# --- 2. 界面美化 ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafd; }
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: -webkit-linear-gradient(#1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .shot-card {
        background: white; padding: 15px; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🎬 AI 视频镜头首尾帧提取器</p>', unsafe_allow_html=True)
st.write("上传视频，AI 将自动识别镜头转换并生成首尾帧对比预览。")

# --- 3. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    threshold = st.slider("识别灵敏度", 10.0, 50.0, 27.0)
    st.info("数值越小越灵敏")

# --- 4. 核心逻辑 ---
uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    
    # 确保环境干净
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始自动化处理"):
        # 每次点击前清空旧目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        with st.spinner("AI 正在扫描镜头并提取画面..."):
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            cap = cv2.VideoCapture(video_path)
            
            if not scene_list:
                st.warning("未检测到明显镜头切换，请调整灵敏度。")
            else:
                # 记录所有处理好的数据
                for i, scene in enumerate(scene_list):
                    start_frame = scene[0].get_frames()
                    end_frame = scene[1].get_frames() - 1
                    
                    st.markdown(f"#### 🎞️ 镜头 {i+1:02d}")
                    
                    # 关键修复：使用 container 确保布局稳定
                    with st.container():
                        cols = st.columns(2)
                        
                        for idx, (f_idx, label) in enumerate([(start_frame, '首帧'), (end_frame, '尾帧')]):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                            ret, frame = cap.read()
                            if ret:
                                # 1. 保存到本地（为了打包 ZIP）
                                img_name = f"shot_{i+1:03d}_{label}.jpg"
                                img_path = os.path.join(output_dir, img_name)
                                cv2.imwrite(img_path, frame)
                                
                                # 2. 在网页显式显示
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                with cols[idx]:
                                    st.image(frame_rgb, caption=f"{label} (第 {f_idx} 帧)", use_container_width=True)
                
                cap.release()
                st.success(f"✅ 处理完成！共提取 {len(scene_list)} 个镜头。")
                st.balloons()

                # 打包下载
                shutil.make_archive("result_frames", 'zip', output_dir)
                with open("result_frames.zip", "rb") as f:
                    st.download_button(
                        label="📥 下载所有截图 (ZIP)",
                        data=f,
                        file_name="shots_archive.zip",
                        mime="application/zip"
                    )
else:
    st.info("请先上传视频文件。")
