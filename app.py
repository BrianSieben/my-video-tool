import streamlit as st
import cv2
import os
import shutil
from scenedetect import detect, ContentDetector

# --- 1. 页面配置与高级美化 ---
st.set_page_config(page_title="VisionShot AI", layout="wide", page_icon="🎬")

# 注入自定义 CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8fafd; font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 3rem !important; font-weight: 800;
        background: -webkit-linear-gradient(#1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    div[data-testid="column"] {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        color: white; border-radius: 8px; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 侧边栏布局 ---
with st.sidebar:
    st.markdown("### 🛠️ 核心参数")
    threshold = st.slider("识别灵敏度", 10.0, 50.0, 27.0)
    st.markdown("---")
    st.caption("VisionShot AI v1.0")

# --- 3. 主界面布局 ---
st.markdown('<p class="main-title">🎬 VisionShot AI</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b;'>智能视频镜头拆解专家</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始分析"):
        with st.spinner("AI 正在扫描镜头..."):
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            cap = cv2.VideoCapture(video_path)
            
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames() - 1
                
                st.markdown(f"#### 🎞️ 镜头 {i+1:02d}")
                cols = st.columns(2)
                
                for idx, (f_idx, label) in enumerate([(start_frame, '开始帧'), (end_frame, '结束帧')]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        img_name = f"shot_{i+1:03d}_{label}.jpg"
                        cv2.imwrite(os.path.join(output_dir, img_name), frame)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cols[idx].image(frame_rgb, caption=f"{label} (第 {f_idx} 帧)")
            
            cap.release()
            st.success("✅ 全部处理完成！")
            st.balloons()

            # --- 打包下载 (重点修复了这里的缩进) ---
            shutil.make_archive("result_frames", 'zip', output_dir)
            with open("result_frames.zip", "rb") as f:
                st.download_button(
                    label="📥 下载所有镜头截图 (ZIP)",
                    data=f,
                    file_name="visionshot_archive.zip",
                    mime="application/zip"
                )
else:
    st.markdown("---")
    st.info("👋 欢迎使用！请在上方上传需要拆解的视频文件。")
