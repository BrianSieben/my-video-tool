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

# --- 核心逻辑 ---
def analyze_image(image_bytes):
    if not model: return {"error": "API Key 未配置"}
    try:
        response = model.generate_content([
            "分析此图并输出JSON：包含风格、内容和氛围维度。仅输出纯JSON。",
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        txt = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(txt)
    except Exception as e:
        return {"error": str(e)}

# --- 界面 ---
st.title("🎬 VisionShot AI Pro")
uploaded_file = st.file_uploader("上传视频", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    if not os.path.exists("output"): os.makedirs("output")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始分析"):
        shutil.rmtree("output", ignore_errors=True)
        os.makedirs("output")
        
        scenes = detect(video_path, ContentDetector(threshold=27.0))
        cap = cv2.VideoCapture(video_path)
        
        for i, scene in enumerate(scenes):
            st.subheader(f"镜头 {i+1}")
            cols = st.columns(2) # 提取首尾两帧
            frames = [scene[0].get_frames(), scene[1].get_frames()-1]
            
            for idx, f_idx in enumerate(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    img_path = f"output/s_{i}_{idx}.jpg"
                    cv2.imwrite(img_path, frame)
                    with cols[idx]:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if st.button(f"🔍 分析/编辑帧 {idx+1}", key=f"ai_{i}_{idx}"):
                            res = analyze_image(open(img_path, "rb").read())
                            if "error" not in res:
                                df = pd.DataFrame(list(res.items()), columns=["维度", "内容"])
                                edited_df = st.data_editor(df, use_container_width=True, key=f"ed_{i}_{idx}")
                                
                                # 导出 Excel
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    edited_df.to_excel(writer, index=False)
                                st.download_button("📥 下载 Excel", output.getvalue(), f"shot_{i+1}.xlsx")
        cap.release()
