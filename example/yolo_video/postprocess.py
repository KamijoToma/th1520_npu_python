import numpy as np
import cv2
# import matplotlib.pyplot as plt

# ================= 配置参数 (需要根据你的模型调整) =================
IMG_SIZE = (384, 640)  # 模型的输入尺寸 (高度, 宽度)
CONF_THRESHOLD = 0.2   # 置信度阈值
IOU_THRESHOLD = 0.7    # NMS的IoU阈值
CLASS_NAMES = open('coco.names', 'r').readlines()        # 替换为你的类别名称列表 (80类COCO)

# YOLOv5n的默认锚框配置 (来自官方yolov5n.yaml)
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],    # P3/8  (stride=8)
    [[30, 61], [62, 45], [59, 119]],   # P4/16 (stride=16)
    [[116, 90], [156, 198], [373, 326]] # P5/32 (stride=32)
]

# ================= 工具函数 =================
def sigmoid_acc(x):
    """ Sigmoid激活函数 """
    return 1 / (1 + np.exp(-x))

def sigmoid(x):
    return x/(2+2*np.abs(x))+0.5

def load_tensor(txt_path, shape):
    """ 从txt文件加载张量数据 """
    data = np.loadtxt(txt_path, dtype=np.float32)
    return data.reshape(shape)

# ================= 核心解码函数 =================
def decode_output(output, anchors, stride):
    """
    解码单个输出层的预测结果
    参数:
        output: 形状为(1, 255, H, W)的numpy数组
        anchors: 当前层的锚框列表 [[w1, h1], [w2, h2], ...]
        stride: 特征图的步长 (8/16/32)
    返回:
        (boxes, scores, class_ids)
    """
    # 转置为(H, W, 255)
    output = output[0].transpose(1, 2, 0)
    h, w = output.shape[:2]
    num_anchors = len(anchors)
    
    # 分割预测值：每个锚框预测85个值 (x,y,w,h,conf + 80类)
    output = output.reshape(h, w, num_anchors, -1)
    
    # 生成网格坐标
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    grid = np.stack((grid_x, grid_y), axis=-1)  # (H, W, 2)
    grid = grid[:, :, np.newaxis, :]  # 扩展维度为 (H, W, 1, 2)
    
    # 解码坐标
    xy = (sigmoid(output[..., 0:2]) * 2 - 0.5 + grid) * stride  # 中心坐标
    wh = (sigmoid(output[..., 2:4]) * 2) ** 2 * anchors  # 宽度高度
    
    # 转换为(x1,y1,x2,y2)格式
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    boxes = np.concatenate((x1y1, x2y2), axis=-1).reshape(-1, 4)
    
    # 处理置信度和类别
    conf = sigmoid(output[..., 4:5]).reshape(-1)
    cls = sigmoid(output[..., 5:]).reshape(-1, 80)
    class_ids = np.argmax(cls, axis=1)
    scores = conf * np.max(cls, axis=1)
    
    return boxes, scores, class_ids

# ================= 主处理流程 =================
def scale_coords(boxes, model_shape, image_shape, padding_info=(0, 0, 1)):
    """根据填充信息将坐标转换到原始图像空间"""
    dx, dy, scale = padding_info
    # 去除填充影响
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
    
    # 限制坐标在图像范围内
    np.clip(boxes[:, 0], 0, image_shape[1], out=boxes[:, 0])  # width
    np.clip(boxes[:, 1], 0, image_shape[0], out=boxes[:, 1])  # height
    np.clip(boxes[:, 2], 0, image_shape[1], out=boxes[:, 2])
    np.clip(boxes[:, 3], 0, image_shape[0], out=boxes[:, 3])
    
    return boxes

def process_outputs(outputs, image_shape, padding_info=(0, 0, 1)):
    """处理所有三个输出层 (添加padding_info参数)"""
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    # 遍历三个输出层 
    for i, (output, anchors) in enumerate(zip(outputs, ANCHORS)):
        # 转换为numpy数组并解码
        boxes, scores, class_ids = decode_output(
            output, 
            np.array(anchors), 
            stride=8*(2**i)  # 8,16,32
        )
        
        # 过滤低置信度
        mask = scores > CONF_THRESHOLD
        all_boxes.append(boxes[mask])
        all_scores.append(scores[mask])
        all_class_ids.append(class_ids[mask])
    
    # 合并结果
    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)
    
    # NMS处理
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(),
        CONF_THRESHOLD, IOU_THRESHOLD
    )
    
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 筛选最终结果
    final_boxes = boxes[indices.flatten()]
    final_scores = scores[indices.flatten()]
    final_class_ids = class_ids[indices.flatten()]
    
    # 坐标转换（新增padding_info参数）
    final_boxes = scale_coords(
        final_boxes, 
        IMG_SIZE, 
        image_shape,
        padding_info  # 传递填充信息
    )
    
    return final_boxes, final_scores, final_class_ids

# ================= 可视化 =================
def plot_results(image, boxes, scores, class_ids):
    """ 绘制检测结果 """
    img_disp = image.copy()
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        print(f"Box: {box} score={score} cls_id={cls_id}")
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制矩形
        color = (0, 255, 0)  # BGR格式
        cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{CLASS_NAMES[cls_id].strip()}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_disp, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_disp, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    
    # 转换颜色空间用于matplotlib显示
    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# ================= 使用示例 =================
if __name__ == "__main__":
    # 1. 加载输出张量
    outputs = [
        load_tensor("output1.txt", (1, 255, 48, 80)),  # 48x80层
        load_tensor("output2.txt", (1, 255, 24, 40)),  # 24x40层
        load_tensor("output3.txt", (1, 255, 12, 20))   # 12x20层
    ]
    
    # 2. 加载原始图像 (用于绘制结果)
    original_image = cv2.imread("kite.jpg")  # 替换为你的图片路径
    image_shape = original_image.shape[:2]  # (高度, 宽度)
    
    # 3. 执行后处理
    boxes, scores, class_ids = process_outputs(outputs, image_shape)
    
    # 4. 绘制结果
    if len(boxes) > 0:
        plot_results(original_image, boxes, scores, class_ids)
    else:
        print("未检测到任何目标")
