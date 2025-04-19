import shlbind as shl # Shl lib
import numpy as np
import ctypes

class Csinn:
    def __init__(self, model_param_path: str):
        '''
        Init Csinn engine with model.
        This function take a lot of time
        '''
        print('Init csinn')
        with open(model_param_path, 'rb') as f:
            file_data = f.read()
            pass
        c_data = ctypes.c_char_p(file_data)
        self.sess = ctypes.cast(shl.csinn_(c_data), ctypes.POINTER(shl.struct_csinn_session))
        self.input_shape = []
        pass

    def __del__(self):
        # Goodbye
        shl.csinn_session_deinit(self.sess)
        shl.csinn_free_session(self.sess)

    def get_input_shape(self) -> list:
        '''
        return the shape of input tensor.
        '''
        if not len(self.input_shape) == 0:
            return self.input_shape
        # 1. 获取输出数量
        input_num = shl.csinn_get_input_number(self.sess)

        # 2. 遍历每个输出
        for i in range(input_num):
            # 初始化输出张量结构体
            input_tensor = shl.csinn_tensor()
            input_tensor.data = ctypes.c_void_p(None)  # 显式初始化为空指针

            # 3. 获取原始输出张量
            shl.csinn_get_input(i, ctypes.pointer(input_tensor), self.sess)

            # 计算元素总数和形状
            dims = [input_tensor.dim[j] for j in range(input_tensor.dim_count)]
            self.input_shape.append(dims)
        return self.input_shape
    
    def eval(self, input: list[np.ndarray]) -> list[np.ndarray]:
        '''
        eval model
        '''
        # 1. 准备输入张量指针数组
        input_tensors = []
        for idx, data_np in enumerate(input):
            # 为每个输入分配张量内存
            tensor = shl.csinn_alloc_tensor(None)

            # 设置维度信息
            tensor.contents.dim_count = len(data_np.shape)
            for i in range(len(data_np.shape)):
                tensor.contents.dim[i] = data_np.shape[i]

            # 将 numpy 数据转换为 C 兼容格式
            data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # 数据类型转换（假设使用索引 0）
            converted_ptr = shl.shl_ref_f32_to_input_dtype(
                ctypes.c_uint32(idx),
                data_ptr,
                self.sess
            )

            # 将转换后的指针绑定到张量
            tensor.contents.data = ctypes.cast(converted_ptr, ctypes.c_void_p)
            input_tensors.append(tensor)

        # 2. 构建输入张量指针数组
        input_array = (ctypes.POINTER(shl.csinn_tensor) * len(input_tensors))()
        for i, tensor in enumerate(input_tensors):
            input_array[i] = tensor
        input_ptr = ctypes.cast(input_array, ctypes.POINTER(ctypes.POINTER(shl.csinn_tensor)))

        # 3. 执行推理
        shl.csinn_update_input_and_run(
            input_ptr,
            ctypes.cast(self.sess, ctypes.POINTER(None))  # 保持与 C 函数的类型兼容
        )

        # 4. 获取输出结果
        outputs = []
        output_num = shl.csinn_get_output_number(self.sess)
        for i in range(output_num):
            output_tensor = shl.csinn_tensor()
            output_tensor.data = ctypes.c_void_p(None)
            shl.csinn_get_output(i, ctypes.byref(output_tensor), self.sess)

            # 转换为 float32 张量
            f32_tensor_ptr = shl.shl_ref_tensor_transform_f32(ctypes.byref(output_tensor))
            f32_tensor = f32_tensor_ptr.contents

            # 转换为 numpy 数组
            if f32_tensor.data:
                dims = [f32_tensor.dim[j] for j in range(f32_tensor.dim_count)]
                total = int(np.prod(dims))
                buf_type = ctypes.c_float * total
                buffer = ctypes.cast(f32_tensor.data, ctypes.POINTER(buf_type))
                np_array = np.ctypeslib.as_array(buffer.contents).reshape(dims).copy()
                outputs.append(np_array)
            else:
                outputs.append(None)

            # 释放转换后的张量
            shl.shl_ref_tensor_transform_free_f32(f32_tensor_ptr)

        # 5. 清理输入张量
        for tensor in input_tensors:
            shl.csinn_free_tensor(tensor)

        return outputs

def prepare_yolo_input(image_path, target_shape=(640, 384)):
    import cv2
    # 1. 读取图像并调整大小
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_shape)  # OpenCV 使用 (width, height)

    # 2. 颜色空间转换 BGR -> RGB（根据模型需要）
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3. 归一化到 [0.0, 1.0] 并转换为 float32
    img_float32 = (img_rgb / 255.0).astype(np.float32)

    # 4. 重塑为 NCHW 格式 (Batch, Channel, Height, Width)
    img_nchw = img_float32.transpose(2, 0, 1)[np.newaxis, ...]  # 添加 batch 维度

    # 5. 确保内存连续
    img_contiguous = np.ascontiguousarray(img_nchw)

    return img_contiguous

def test():
    import cv2
    from postprocess import process_outputs, CLASS_NAMES

    eng = Csinn('./model.params')
    print(eng.get_input_shape())
    img = prepare_yolo_input('kite.jpg')
    output = eng.eval([img, ])
    print(f'output shape: {repr([i.shape for i in output])}')
    # 2. 加载原始图像 (用于绘制结果)
    original_image = cv2.imread("kite.jpg")  # 替换为你的图片路径
    image_shape = original_image.shape[:2]  # (高度, 宽度)

    # 3. 执行后处理
    boxes, scores, class_ids = process_outputs(output, image_shape)

    # 4. 绘制结果
    for box, score, cls_id in zip(boxes, scores, class_ids):
        print(f"Box: {box} score={score} cls_id={cls_id} class={CLASS_NAMES[cls_id]}")

if __name__ == '__main__':
    test()