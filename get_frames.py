import os
import cv2
import numpy as np
from scenedetect import detect, ContentDetector

def extract_and_combine(video_path, output_dir='output_frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"🚀 开始分析视频并识别镜头...")
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    
    cap = cv2.VideoCapture(video_path)
    collected_images = [] # 用于存放我们要拼接的图片

    print(f"🎬 识别到 {len(scene_list)} 个镜头，正在处理...")
    
    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames() - 1 

        for frame_idx, label in [(start_frame, 'start'), (end_frame, 'end')]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 1. 保存单张图片
                filename = f"shot_{i+1:03d}_{label}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                
                # 2. 为拼接做准备：统一缩小尺寸以免大图太大
                small_frame = cv2.resize(frame, (320, 180)) # 缩小为 320x180
                # 在图上画出是哪个镜头
                cv2.putText(small_frame, f"Shot {i+1} {label}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                collected_images.append(small_frame)
    
    cap.release()

    # --- 拼接长图逻辑 ---
    if collected_images:
        print(f"🖼️ 正在生成总预览长图...")
        # 每行放 4 张图
        cols = 4
        rows = (len(collected_images) + cols - 1) // cols
        
        # 补齐空白格子，防止拼图失败
        while len(collected_images) < rows * cols:
            collected_images.append(np.zeros_like(collected_images[0]))
        
        # 拼成矩阵
        row_images = []
        for r in range(rows):
            row_images.append(np.hstack(collected_images[r*cols : (r+1)*cols]))
        final_image = np.vstack(row_images)
        
        # 保存总图
        cv2.imwrite("final_storyboard.jpg", final_image)
        print(f"✅ 大功告成！总图已保存为: final_storyboard.jpg")

if __name__ == "__main__":
    # 记得改成你自己的视频文件名
    video_name = "首尾帧测试123.mp4" 
    if os.path.exists(video_name):
        extract_and_combine(video_name)
    else:
        print(f"❌ 找不到视频文件")