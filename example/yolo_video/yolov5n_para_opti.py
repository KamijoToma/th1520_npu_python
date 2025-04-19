import multiprocessing as mp
import numpy as np
import cv2
import time
from queue import Empty
from shllib import Csinn
from postprocess import process_outputs, CLASS_NAMES

class ParallelVideoProcessor:
    def __init__(self, model_path, input_size=(640, 384), num_post_workers=2, num_pre_workers=2):
        self.model_path = model_path
        self.input_size = input_size
        self.num_post_workers = num_post_workers
        self.num_pre_workers = num_pre_workers
        
        # 共享队列配置
        self.input_queue = mp.Queue(maxsize=4)      # 原始帧队列
        self.preprocess_queue = mp.Queue(maxsize=8) # 预处理队列
        self.infer_queue = mp.Queue(maxsize=2)      # 推理结果队列
        self.output_queue = mp.Queue(maxsize=8)     # 后处理结果队列

    def _frame_reader(self, input_path):
        """视频读取进程：负责读取原始帧并分配序号"""
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 发送终止信号： (None, None, None)
            self.input_queue.put((frame_idx, frame, time.time()))
            frame_idx += 1
        self.input_queue.put((None, None, None))  # 结束信号
        cap.release()

    def _preprocess_worker(self, worker_id):
        '''Preprocess'''
        while True:
            try:
                # 获取带序号的帧数据
                frame_id, frame, timestamp = self.input_queue.get(timeout=60)
                if frame_id is None:  # 终止信号
                    self.preprocess_queue.put((None, None, None))
                    break
                input_data, orig_size, padding_info = self._preprocess(frame)
                self.preprocess_queue.put(((frame_id, frame, timestamp), input_data, (orig_size, padding_info)))
            except Empty:
                print(f'Preprocess worker {worker_id} exited')

    def _inference_worker(self):
        """推理进程：单进程运行模型"""
        model = Csinn(self.model_path)
        while True:
            try:
                # 获取带序号的帧数据
                p1, input_data, p3 = self.preprocess_queue.get(timeout=30)
                if p1 is None:  # 终止信号
                    self.infer_queue.put((None, None, None, None))
                    break

                frame_id, frame, timestamp = p1
                orig_size, padding_info = p3
                outputs = model.eval([input_data])
                
                # 传递原始帧 + 推理结果
                self.infer_queue.put((
                    frame_id, 
                    frame, 
                    outputs, 
                    (orig_size, padding_info, timestamp)
                ))
            except Empty:
                break

    def _postprocess_worker(self, worker_id):
        """后处理进程：多实例并行"""
        while True:
            try:
                data = self.infer_queue.get(timeout=120) # Set a longer timeout due to long load time of Csinn
                if data[0] is None:  # 终止信号
                    self.output_queue.put((None, None, None))
                    break
                
                frame_id, frame, outputs, params = data
                orig_size, padding_info, timestamp = params
                
                # 后处理
                boxes, scores, class_ids = process_outputs(
                    outputs, 
                    orig_size, 
                    padding_info
                )
                
                # 传递处理结果
                self.output_queue.put((
                    frame_id, 
                    self._draw_detections(frame, boxes, scores, class_ids), 
                    timestamp
                ))
            except Empty:
                print(f'Post worker {worker_id} stopped')
                break

    def _writer_worker(self, output_path, total_frames):
        """写入进程：保证输出顺序"""
        input_q = []
        inf_q = []
        out_q = []
        pre_q = []
        expected_id = 0
        buffer = {}
        t1 = 0
        t2 = 0
        
        while expected_id < total_frames:
            # queue profiling
            frame_id, frame, _ = self.output_queue.get()
            input_q.append(self.input_queue.qsize())
            inf_q.append(self.infer_queue.qsize())
            out_q.append(self.output_queue.qsize())
            pre_q.append(self.preprocess_queue.qsize())
            # 获取带序号的结果
            if frame_id == 0:
                t1 = time.time()
            if frame_id == expected_id:
                cv2.imwrite(f'output/f{frame_id}.jpg', frame)
                print(f'Write {expected_id}/{total_frames}')
                expected_id +=1
                # 检查缓冲中是否有后续帧
                while expected_id in buffer:
                    cv2.imwrite(f'output/f{frame_id}.jpg', frame)
                    expected_id +=1
            elif frame_id > expected_id:
                buffer[frame_id] = frame
        t2 = time.time()
        print(f"Time spend: {t2-t1} = {total_frames/(t2-t1)}fps")
        print(f'Infq: {sum(inf_q)/len(inf_q)}')
        print(f'Inputq: {sum(input_q)/len(input_q)}')
        print(f'OutQ: {sum(out_q)/len(out_q)}')
        print(f'PreQ: {sum(pre_q)/len(pre_q)}')
        pass

    def process_video(self, input_path, output_path):
        # 启动所有工作进程
        processes = []
        
        # 视频读取进程
        p_reader = mp.Process(target=self._frame_reader, args=(input_path,))
        processes.append(p_reader)
        
        # 推理进程（仅1个）
        p_infer = mp.Process(target=self._inference_worker)
        processes.append(p_infer)
        
        # 后处理进程池
        post_workers = [
            mp.Process(target=self._postprocess_worker, args=(i,))
            for i in range(self.num_post_workers)
        ]
        processes.extend(post_workers)

        # 预处理进程池
        pre_workers = [
            mp.Process(target=self._preprocess_worker, args=(i, ))
            for i in range(self.num_pre_workers)
        ]
        processes.extend(pre_workers)
        
        # 启动所有进程
        for p in processes:
            p.start()
        
        # 主进程负责写入
        total_frames = int(cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FRAME_COUNT))
        self._writer_worker(output_path, total_frames)
        
        # 清理
        for p in processes:
            p.join()


    def _preprocess(self, frame):
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
    
    def _draw_detections(self, frame, boxes, scores, class_ids):
        """在原始帧上绘制检测结果"""
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES[cls_id]
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

if __name__ == "__main__":
    detector = ParallelVideoProcessor(
        model_path="./model.params",
        input_size=(640, 384)  # 与模型输入尺寸一致
    )
    
    # 处理示例视频
    detector.process_video(
        input_path="input.mp4",
        output_path="./output_video.avi"
    )