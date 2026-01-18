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

# 页面基础配置
st.set_page_config(page_title="VisionShot AI Pro", layout="wide")

# --- 1. 初始化 Gemini (从 Secrets 读取) ---
def get_gemini_model():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ 未找到 API Key，请在 Settings -> Secrets 中配置 GEMINI_API_KEY")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Gemini 初始化失败: {str(e)}")
        return None

model = get_gemini_model()

# --- 2. 界面展示 ---
st.title("🎬 VisionShot AI Pro")
st.info("提示：如果遇到分析失败，请检查 API Key 额度或网络连接。")

uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "mov", "avi"])

if uploaded_file:
    # 临时存放视频和截图
    video_path = "temp_video.mp4"
    output_dir = "frames_cache"
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 开始提取并分析镜头"):
        # 清理旧缓存
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        with st.spinner("正在智能识别镜头..."):
            # 识别场景
            scenes = detect(video_path, ContentDetector(threshold=27.0))
            cap = cv2.VideoCapture(video_path)
            
            for i, scene in enumerate(scenes):
                st.markdown(f"### 🎞️ 镜头 {i+1:02d}")
                cols = st.columns(2) # 提取首尾两帧
                
                # 获取首帧和尾帧的索引
                frames_to_capture = [scene[0].get_frames(), scene[1].get_frames() - 1]
                
                for idx, f_idx in enumerate(frames_to_capture):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        img_name = f"shot_{i+1}_{idx}.jpg"
                        img_path = os.path.join(output_dir, img_name)
                        cv2.imwrite(img_path, frame)
                        
                        with cols[idx]:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"帧 {idx+1}")
                            
                            # AI 分析按钮
                            if st.button(f"🔍 分析此帧并编辑", key=f"btn_{i}_{idx}"):
                                if model:
                                    with st.spinner("AI 正在解析视觉风格..."):
                                        try:
                                            # 读取图片数据
                                            with open(img_path, "rb") as im_file:
                                                img_data = im_file.read()
                                            
                                            # 发送给 Gemini
                                            response = model.generate_content([
                                                "请以专业视角分析此图的构图、影调和氛围，输出JSON格式。只需输出JSON。",
                                                {"mime_type": "image/jpeg", "data": img_data}
                                            ])
                                            
                                            # 清洗并解析 JSON
                                            clean_txt = response.text.replace('```json', '').replace('```', '').strip()
                                            res_data = json.loads(clean_txt)
                                            
                                            # 生成可编辑表格
                                            df = pd.DataFrame(list(res_data.items()), columns=["维度", "内容"])
                                            edited_df = st.data_editor(df, use_container_width=True, key=f"edit_{i}_{idx}")
                                            
                                            # 导出 Excel
                                            buffer = io.BytesIO()
                                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                                edited_df.to_excel(writer, index=False)
                                            
                                            st.download_button(
                                                label="📥 下载修改后的分析报告 (Excel)",
                                                data=buffer.getvalue(),
                                                file_name=f"shot_{i+1}_analysis.xlsx",
                                                mime="application/vnd.ms-excel"
                                            )
                                        except Exception as ai_err:
                                            st.error(f"AI 分析出错: {ai_err}")
            cap.release()
