import streamlit as st
import cv2
import os
import shutil
from scenedetect import detect, ContentDetector

# --- 1. 页面配置 ---
st.set_page_config(page_title="VisionShot AI", layout="wide", page_icon="🎬")

# 注入 CSS 确保 UI 美观
st.markdown("""
    <style>
    .stApp { background-color: #f8fafd; }
    .main-title {
        font-size: 3rem !important; font-weight: 800;
        background: -webkit-linear-gradient(#1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .shot-container {
        background: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 标题与上传 ---
st.markdown('<p class="main-title">🎬 VisionShot AI</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748b;'>智能视频镜头拆解专家</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])

# --- 3. 核心逻辑 ---
if uploaded_file:
    # 视频保存路径
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    
    # 写入上传的文件
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 按钮点击逻辑
    if st.button("🚀 开始分析", key="run_analysis"):
        # 清理旧目录并新建
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 进度条提示
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # 1. 镜头检测
            progress_text.text("🔍 正在识别镜头切换点...")
            scene_list = detect(video_path, ContentDetector(threshold=27.0))
            
            if not scene_list:
                st.warning("未检测到明显的镜头切换，请尝试上传更复杂的视频。")
            else:
                cap = cv2.VideoCapture(video_path)
                total_scenes = len(scene_list)
                
                # 2. 循环提取帧
                for i, scene in enumerate(scene_list):
                    start_frame = scene[0].get_frames()
                    end_frame = scene[1].get_frames() - 1
                    
                    st.markdown(f"##### 🎞️ 镜头 {i+1:02d}")
                    cols = st.columns(2)
                    
                    for idx, (f_idx, label) in enumerate([(start_frame, '开始帧'), (end_frame, '结束帧')]):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, frame = cap.read()
                        if ret:
                            # 保存
                            img_name = f"shot_{i+1:03d}_{label}.jpg"
                            img_path = os.path.join(output_dir, img_name)
                            cv2.imwrite(img_path, frame)
                            
                            # 显示
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with cols[idx]:
                                st.image(frame_rgb, caption=f"{label} (第 {f_idx} 帧)")
                    
                    # 更新进度
                    progress_bar.progress((i + 1) / total_scenes)
                
                cap.release()
                progress_text.text("✅ 分析完成！")
                st.balloons()

                # 3. 打包下载
                shutil.make_archive("result_frames", 'zip', output_dir)
                with open("result_frames.zip", "rb") as f:
                    st.download_button(
                        label="📥 下载所有镜头截图 (ZIP)",
                        data=f,
                        file_name="visionshot_archive.zip",
                        mime="application/zip"
                    )
        except Exception as e:
            st.error(f"分析过程中出错: {e}")
else:
    st.info("💡 请先上传视频，然后点击“开始分析”按钮。")
