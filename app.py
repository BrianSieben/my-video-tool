import streamlit as st
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import google.generativeai as genai
import json
import io
from scenedetect import detect, ContentDetector

st.set_page_config(page_title="VisionShot AI Pro", layout="wide")

# --- 初始化 Gemini ---
def init_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    return None

model = init_gemini()

# --- 核心界面 ---
st.title("🎬 VisionShot AI Pro")
uploaded_file = st.file_uploader("上传视频", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    if not os.path.exists("output"): os.makedirs("output")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始拆解并分析"):
        shutil.rmtree("output", ignore_errors=True)
        os.makedirs("output")
        
        with st.spinner("正在提取关键帧..."):
            scenes = detect(video_path, ContentDetector(threshold=27.0))
            cap = cv2.VideoCapture(video_path)
            
            for i, scene in enumerate(scenes):
                st.subheader(f"镜头 {i+1}")
                cols = st.columns(2)
                # 提取每个镜头的首尾帧
                frame_indices = [scene[0].get_frames(), scene[1].get_frames()-1]
                
                for idx, f_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        img_path = f"output/s_{i}_{idx}.jpg"
                        cv2.imwrite(img_path, frame)
                        with cols[idx]:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            if st.button(f"🔍 AI分析帧 {idx+1}", key=f"ai_{i}_{idx}"):
                                # Gemini 逻辑
                                try:
                                    img_data = open(img_path, "rb").read()
                                    response = model.generate_content([
                                        "请分析此图并输出JSON（包含风格、构图、色调）。只输出纯JSON。",
                                        {"mime_type": "image/jpeg", "data": img_data}
                                    ])
                                    res_json = json.loads(response.text.replace('```json', '').replace('```', '').strip())
                                    
                                    # 表格显示与编辑
                                    df = pd.DataFrame(list(res_json.items()), columns=["维度", "描述"])
                                    edited_df = st.data_editor(df, use_container_width=True, key=f"edt_{i}_{idx}")
                                    
                                    # Excel 导出
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        edited_df.to_excel(writer, index=False)
                                    st.download_button("📥 下载 Excel", output.getvalue(), f"shot_{i+1}_f{idx}.xlsx")
                                except Exception as e:
                                    st.error(f"分析失败: {e}")
            cap.release()
