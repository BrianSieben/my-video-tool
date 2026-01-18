import streamlit as st
import cv2
import os
import shutil
from scenedetect import detect, ContentDetector

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
    </style>
    """, unsafe_allow_html=True)

# --- 2. 侧边栏 ---
with st.sidebar:
    st.markdown("### 🛠️ 核心参数")
    threshold = st.slider("识别灵敏度", 10.0, 50.0, 27.0, help="数值越小，识别出的镜头越多")
    st.markdown("---")
    st.markdown("#### 功能说明")
    st.write("1. 自动识别镜头切换\n2. 每个镜头提取 4 帧\n3. 支持打包下载 ZIP")
    st.caption("VisionShot AI v1.1")

# --- 3. 主界面布局 ---
st.markdown('<p class="main-title">🎬 VisionShot AI Pro</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>智能视频镜头拆解专家 - 现已支持每个镜头提取 4 帧预览</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("请上传视频文件", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    
    # 初始化文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始深度分析"):
        # 每次点击分析前清空旧数据
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        with st.spinner("AI 正在扫描并计算关键帧..."):
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            cap = cv2.VideoCapture(video_path)
            
            if not scene_list:
                st.warning("未能识别出明显的镜头切换，请调低灵敏度后再试。")
            else:
                # 遍历每个镜头
                for i, scene in enumerate(scene_list):
                    start_frame = scene[0].get_frames()
                    end_frame = scene[1].get_frames() - 1
                    duration = end_frame - start_frame
                    
                    # 计算 4 个关键点的帧索引（首帧、1/3处、2/3处、尾帧）
                    mid_1 = start_frame + int(duration * 0.33)
                    mid_2 = start_frame + int(duration * 0.66)
                    
                    st.markdown(f'<div class="shot-header">🎞️ 镜头 {i+1:02d}</div>', unsafe_allow_html=True)
                    
                    # 建立 4 列布局
                    cols = st.columns(4)
                    
                    # 提取计划
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
                            # 1. 保存到本地用于 ZIP
                            img_name = f"shot_{i+1:03d}_{idx}_{label}.jpg"
                            img_path = os.path.join(output_dir, img_name)
                            cv2.imwrite(img_path, frame)
                            
                            # 2. 转换为 RGB 在网页显示
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with cols[idx]:
                                st.image(frame_rgb, caption=f"{label} (F:{f_idx})", use_container_width=True)
                
                cap.release()
                st.success(f"✅ 处理完成！共分析出 {len(scene_list)} 个镜头。")
                st.balloons()

                # --- 打包下载逻辑 ---
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
