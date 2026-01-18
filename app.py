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

# --- 1. 页面配置 ---
st.set_page_config(page_title="VisionShot AI Pro", layout="wide", page_icon="🎬")

# --- 2. Gemini 初始化 ---
def init_gemini():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Gemini 初始化失败: {e}")
    return None

model = init_gemini()

# --- 3. 核心功能 ---
def analyze_image(image_bytes):
    if not model: return {"error": "API Key 未配置"}
    prompt = "请作为专家分析此图，输出 JSON 格式，包含视觉、内容和氛围维度。仅输出纯 JSON。"
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        txt = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(txt)
    except Exception as e:
        return {"error": str(e)}

def flatten_dict(d, parent_key='', sep=' -> '):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, f"{new_key}{sep}", sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)

# --- 4. 界面展示 ---
st.title("🎬 VisionShot AI Pro")
uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "mov", "avi"])

if uploaded_file:
    video_path = "temp_video.mp4"
    output_dir = "output_frames"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始分析"):
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        scene_list = detect(video_path, ContentDetector(threshold=27.0))
        cap = cv2.VideoCapture(video_path)
        
        for i, scene in enumerate(scene_list):
            start = scene[0].get_frames()
            end = scene[1].get_frames() - 1
            duration = end - start
            st.subheader(f"🎞️ 镜头 {i+1:02d}")
            
            cols = st.columns(4)
            points = [(start, '首帧'), (start+int(duration*0.33), '中1'), (start+int(duration*0.66), '中2'), (end, '尾帧')]
            
            for idx, (f_idx, label) in enumerate(points):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    img_path = os.path.join(output_dir, f"shot_{i+1}_{idx}.jpg")
                    cv2.imwrite(img_path, frame)
                    with cols[idx]:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=label)
                        if st.button(f"🔍 分析/编辑", key=f"ai_{i}_{idx}"):
                            res = analyze_image(open(img_path, "rb").read())
                            if "error" not in res:
                                flat = flatten_dict(res)
                                df = pd.DataFrame(list(flat.items()), columns=["维度", "内容"])
                                # 可编辑表格
                                edited_df = st.data_editor(df, use_container_width=True, key=f"ed_{i}_{idx}")
                                # 导出 Excel
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    edited_df.to_excel(writer, index=False)
                                st.download_button("📥 下载 Excel", output.getvalue(), f"shot_{i+1}.xlsx")
                            else:
                                st.error(res["error"])
        cap.release()
