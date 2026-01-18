import streamlit as st
import cv2
import os
import shutil
from scenedetect import detect, ContentDetector

# --- 1. 页面配置与高级美化 ---
st.set_page_config(page_title="VisionShot AI", layout="wide", page_icon="🎬")

st.markdown("""
    <style>
    .stApp { background-color: #f8fafd; font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 3rem !important; font-weight: 800;
        background: -webkit-linear-gradient(#1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    /* 图片容器样式 */
    .img-box {
        background: white; padding: 10px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 侧边栏 ---
with st.sidebar:
    st.markdown("### 🛠️ 核心参数")
    threshold = st.slider("识别灵敏度", 10.0, 50.0, 27.0)
    st.markdown("---")
    st.caption("VisionShot AI v1.0")

# --- 3. 主界面 ---
st.markdown('<p class="main-title">🎬 VisionShot AI</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b;'>智能视频镜头拆解专家</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    
    # 初始化环境
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始分析"):
        # 每次点击分析前清空旧图片
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        with st.spinner("AI 正在扫描镜头..."):
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            cap = cv2.VideoCapture(video_path)
            
            # --- 关键修复：确保在循环中立即展示图片 ---
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames() - 1
                
                st.markdown(f"##### 🎞️ 镜头 {i+1:02d}")
                cols = st.columns(2)
                
                # 定义需要抓取的帧
                for idx, (f_idx, label) in enumerate([(start_frame, '开始帧'), (end_frame, '结束帧')]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        # 保存到本地（为了打包下载）
                        img_name = f"shot_{i+1:03d}_{label}.jpg"
                        img_path = os.path.join(output_dir, img_name)
                        cv2.imwrite(img_path, frame)
                        
                        # 在网页实时渲染
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with cols[idx]:
                            st.image(frame_rgb, caption=f"{label} (第 {f_idx} 帧)", use_container_width=True)
            
            cap.release()
            
        if len(scene_list) > 0:
            st.success(f"✅ 处理完成！共提取 {len(scene_list)} 个场景。")
            st.balloons()

            # --- 一键打包下载 ---
            shutil.make_archive("result_frames", 'zip', output_dir)
            with open("result_frames.zip", "rb") as f:
                st.download_button(
                    label="📥 下载所有镜头截图 (ZIP)",
                    data=f,
                    file_name="visionshot_archive.zip",
                    mime="application/zip"
                )
else:
    st.info("👋 欢迎使用！请在上方上传视频文件开始分析。")
