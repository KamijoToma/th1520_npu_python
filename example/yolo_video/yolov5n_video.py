import cv2
import numpy as np
from shllib import Csinn  # 假设您的类保存在这个文件
from postprocess import process_outputs, CLASS_NAMES

class VideoDetector:
    def __init__(self, model_path, input_size=(640, 384)):
        self.model = Csinn(model_path)
        self.input_size = input_size  # (width, height)
        self.class_names = CLASS_NAMES
        
    def preprocess(self, frame):
        """保持宽高比的缩放和填充"""
        # 原始尺寸
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(frame, (new_w, new_h))
        
        # 创建画布并填充
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
        
        # 转换到模型输入格式
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_float = (img_rgb / 255.0).astype(np.float32)
        img_nchw = img_float.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(img_nchw), (w, h), (dx, dy, scale)

    def postprocess(self, outputs, orig_size, padding_info):
        """将检测结果转换回原始坐标"""
        boxes, scores, class_ids = process_outputs(outputs, orig_size[::-1])
        
        # 去除填充影响
        dx, dy, scale = padding_info
        if boxes.size > 0:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
            
            # 限制坐标在图像范围内
            np.clip(boxes[:, 0], 0, orig_size[0], out=boxes[:, 0])
            np.clip(boxes[:, 1], 0, orig_size[1], out=boxes[:, 1])
            np.clip(boxes[:, 2], 0, orig_size[0], out=boxes[:, 2])
            np.clip(boxes[:, 3], 0, orig_size[1], out=boxes[:, 3])
        
        return boxes, scores, class_ids

    def draw_detections(self, frame, boxes, scores, class_ids):
        """在原始帧上绘制检测结果"""
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[cls_id]
            color = (0, 255, 0)  # BGR格式
            
            # 绘制矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{class_name}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            
            # 绘制文本
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return frame

    def process_video(self, input_path, output_path):
        """处理视频主函数"""
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
        
        # 获取视频信息
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
        # gst_str = 'appsrc ! videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4'
        #writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480), apiPreference=cv2.CAP_FFMPEG)
        # out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (640, 480))
        
        # 处理每一帧
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预处理
            input_data, orig_size, padding_info = self.preprocess(frame)
            
            # 推理
            outputs = self.model.eval([input_data])
            
            # 后处理
            boxes, scores, class_ids = self.postprocess(outputs, (orig_width, orig_height), padding_info)
            
            # 绘制结果
            result_frame = self.draw_detections(frame.copy(), boxes, scores, class_ids)
            
            # 写入输出
            cv2.imwrite(f'output/f{frame_count}.jpg', result_frame)
            
            # 进度显示
            frame_count += 1
            print(f"\r处理进度: {frame_count}/{total_frames} ({frame_count/total_frames:.1%})", end="")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n处理完成!")

if __name__ == "__main__":
    detector = VideoDetector(
        model_path="./model.params",
        input_size=(640, 384)  # 与模型输入尺寸一致
    )
    
    # 处理示例视频
    detector.process_video(
        input_path="input.mp4",
        output_path="output_video.mp4"
    )