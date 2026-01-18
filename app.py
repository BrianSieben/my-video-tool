import streamlit as st
import cv2
import os
import shutil
from scenedetect import detect, ContentDetector

# --- 页面配置 ---
st.set_page_config(page_title="AI 视频镜头实验室", layout="wide", page_icon="🎬")

# --- 标题 ---
st.title("🎬 AI 视频镜头首尾帧提取器")
st.write("上传视频，AI 将自动识别镜头转换并生成首尾帧对比预览。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    threshold = st.slider("识别灵敏度", 10.0, 50.0, 27.0)
    st.info("数值越小越灵敏")

# --- 文件上传 ---
uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "mov", "avi"])

if uploaded_file:
    # 建立本地临时文件夹
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # 清空旧的
    os.makedirs(output_dir)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始自动化处理"):
        with st.spinner("AI 正在扫描镜头..."):
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            cap = cv2.VideoCapture(video_path)
            
            for i, scene in enumerate(scene_list):
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames() - 1
                
                # 展示网格
                st.markdown(f"### 🎞️ 镜头 {i+1}")
                cols = st.columns(2)
                
                for idx, (f_idx, label) in enumerate([(start_frame, 'start'), (end_frame, 'end')]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        # 保存本地
                        img_name = f"shot_{i+1:03d}_{label}.jpg"
                        cv2.imwrite(os.path.join(output_dir, img_name), frame)
                        # 网页展示
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cols[idx].image(frame_rgb, caption=f"{label} 帧")
            
            cap.release()
            st.success("✅ 处理完成！")
            st.balloons()

            # --- 一键打包下载 ---
            shutil.make_archive("result_frames", 'zip', output_dir)
            with open("result_frames.zip", "rb") as f:
                st.download_button(
                    label="📥 下载所有截图 (ZIP)",
                    data=f,
                    file_name="shots_archive.zip",
                    mime="application/zip"
                )
