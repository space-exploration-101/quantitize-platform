#!/usr/bin/env python3
"""
简单的 ONNX 模型推理脚本
使用 Python NMS 处理检测结果，并绘制检测框

输入：图片路径和 ONNX 模型路径
输出：带有检测框和识别结果的图片
"""

import argparse
import os
import tempfile
import shutil
import cv2
import numpy as np
import _pin_stdlib_platform  # noqa: F401, E402

import onnxruntime as ort
import torch
import sys


def _imread_unicode(path: str):
    """支持中文路径的图片读取（Windows 下 cv2.imread 可能失败）"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _imwrite_unicode(path: str, img: np.ndarray):
    """支持中文路径的图片保存（先写临时文件再移动，避免 Windows 下损坏）"""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower() or '.png'
    if ext not in ['.png', '.jpg', '.jpeg']:
        ext = '.png'
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise ValueError("图片编码失败")
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix='img_')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(buf.tobytes())
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def get_model_input_info(onnx_model_path, session):
    """获取模型输入信息"""
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    input_type = model_inputs[0].type
    
    # 从输入形状中提取期望的尺寸
    if len(input_shape) == 4:
        model_input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
        model_input_width = input_shape[3] if isinstance(input_shape[3], int) else 640
    else:
        model_input_height = 640
        model_input_width = 640
    
    return model_input_height, model_input_width, input_type, input_name


def preprocess_image(img, target_size, use_grayscale=False, grayscale_ch0_only=False):
    """
    预处理图片：resize 并转换为模型输入格式
    
    Args:
        img: OpenCV 读取的图片 (BGR格式)
        target_size: 目标尺寸 (height, width)
        use_grayscale: 若为 True，转为黑白图
        grayscale_ch0_only: 若为 True（且 use_grayscale 为 True），仅第一通道为灰度值，第二、三通道为 0
    
    Returns:
        img_tensor: 预处理后的张量 (1, 3, H, W), RGB, float32, 0-1范围
        orig_img_shape: 原始图片尺寸 (H, W)
    """
    orig_img_shape = img.shape[:2]  # (H, W)
    
    # Resize 图片到目标尺寸
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
    
    if use_grayscale:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        if grayscale_ch0_only:
            # 第一通道=灰度，第二、三通道=0
            img_rgb = np.zeros((*gray.shape, 3), dtype=gray.dtype)
            img_rgb[..., 0] = gray
        else:
            # 三通道 (H, W, 3)，每通道值相同
            img_rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        # BGR to RGB
        img_rgb = img_resized[..., ::-1]
    
    # HWC to CHW
    img_tensor = img_rgb.transpose(2, 0, 1)
    
    # 添加 batch 维度: (1, 3, H, W)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # 归一化到 [0, 1] 并转换为 float32
    img_tensor = img_tensor.astype(np.float32) / 255.0
    
    return img_tensor.astype(np.float16), orig_img_shape


def run_onnx_inference(onnx_model_path, img_tensor, input_name):
    """
    运行 ONNX 推理
    
    Args:
        onnx_model_path: ONNX 模型路径
        img_tensor: 预处理后的图片张量
        input_name: 输入节点名称
    
    Returns:
        prediction: 模型输出 (batch, num_features, num_boxes)
    """
    # 创建 ONNX Runtime 会话（优先使用 GPU）
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # 运行推理
    outputs = session.run(None, {input_name: img_tensor})
    prediction = outputs[0]
    
    # 确保 prediction 是 (batch, num_features, num_boxes) 格式
    if len(prediction.shape) == 2:
        prediction = np.expand_dims(prediction, axis=0)
    
    # 转换为 float32
    prediction = prediction.astype(np.float32)
    
    return prediction


def run_python_nms(prediction, conf_thres=0.25, iou_thres=0.7, max_det=300, nc=2):
    """
    使用 Python NMS 处理检测结果
    
    Args:
        prediction: 模型输出 (batch, num_features, num_boxes)
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        max_det: 最大检测数量
        nc: 类别数量
    
    Returns:
        detections: 检测结果 (N, 6) 格式: [x1, y1, x2, y2, conf, cls]
    """
    # 尝试从 ultralytics 导入 NMS
    try:
        # 尝试从项目中的 ultralytics 导入
        train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../train'))
        if train_path not in sys.path:
            sys.path.insert(0, train_path)
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        try:
            # 尝试从已安装的 ultralytics 导入
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            raise ImportError("无法导入 non_max_suppression，请确保 ultralytics 已安装或路径正确")
    
    # 转换为 torch tensor
    prediction_torch = torch.from_numpy(prediction).cpu()
    
    # 运行 NMS
    outputs = non_max_suppression(
        prediction_torch,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        nc=nc,
    )
    
    # 转换为 numpy array
    if len(outputs) > 0 and len(outputs[0]) > 0:
        detections = outputs[0].numpy()
    else:
        detections = np.array([])
    
    return detections


def scale_boxes(boxes, prep_img_shape, orig_img_shape):
    """
    将边界框坐标从预处理后的尺寸转换回原始图片尺寸
    
    Args:
        boxes: 边界框坐标 (N, 4)，格式为 xyxy
        prep_img_shape: 预处理后的图片尺寸 (height, width)
        orig_img_shape: 原始图片尺寸 (height, width)
    
    Returns:
        转换后的边界框坐标 (N, 4)
    """
    if len(boxes) == 0:
        return boxes
    
    orig_h, orig_w = orig_img_shape[0], orig_img_shape[1]
    prep_h, prep_w = prep_img_shape[0], prep_img_shape[1]
    
    # 计算缩放比例
    gain = min(prep_h / orig_h, prep_w / orig_w)
    
    # 计算 padding
    pad_x = round((prep_w - orig_w * gain) / 2 - 0.1)
    pad_y = round((prep_h - orig_h * gain) / 2 - 0.1)
    
    # 复制 boxes 以避免修改原始数据
    boxes_scaled = boxes.copy()
    
    # 减去 padding
    boxes_scaled[:, 0] -= pad_x  # x1
    boxes_scaled[:, 1] -= pad_y  # y1
    boxes_scaled[:, 2] -= pad_x  # x2
    boxes_scaled[:, 3] -= pad_y  # y2
    
    # 除以缩放比例
    boxes_scaled[:, :4] /= gain
    
    # 裁剪到图片范围内
    boxes_scaled[:, [0, 2]] = boxes_scaled[:, [0, 2]].clip(0, orig_w)  # x1, x2
    boxes_scaled[:, [1, 3]] = boxes_scaled[:, [1, 3]].clip(0, orig_h)  # y1, y2
    
    return boxes_scaled


def draw_detections(img, detections):
    """
    在图片上绘制检测框和标签
    
    Args:
        img: 原始图片 (BGR格式)
        detections: 检测结果 (N, 6) 格式: [x1, y1, x2, y2, conf, cls]
    
    Returns:
        result_img: 绘制了检测框的图片
    """
    result_img = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det[0:4]
        conf = det[4]
        cls = int(det[5])
        
        # 转换为整数并确保坐标在图片范围内
        x1 = max(0, min(int(x1), img.shape[1]))
        y1 = max(0, min(int(y1), img.shape[0]))
        x2 = max(0, min(int(x2), img.shape[1]))
        y2 = max(0, min(int(y2), img.shape[0]))
        
        # 绘制边界框（绿色）
        color = (0, 255, 0)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景和文字
        label = f"Class {cls} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(result_img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_img


def inference_with_python_nms(onnx_model_path, image_path, output_path=None, 
                              conf_thres=0.25, iou_thres=0.7, max_det=300, nc=2, use_grayscale=False, grayscale_ch0_only=False):
    """
    使用 ONNX 模型和 Python NMS 进行推理
    
    Args:
        onnx_model_path: ONNX 模型路径
        image_path: 输入图片路径
        output_path: 输出图片路径（如果为 None，则自动生成）
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        max_det: 最大检测数量
        nc: 类别数量
        use_grayscale: 若为 True，将输入转为黑白图再推理
        grayscale_ch0_only: 若为 True，仅第一通道为灰度值，第二、三通道为 0（需同时 use_grayscale=True）
    
    Returns:
        output_path: 输出图片路径
    """
    print("=" * 60)
    print("ONNX 推理 + Python NMS")
    print("=" * 60)
    
    # 1. 加载图片
    print(f"\n1. 加载图片: {image_path}")
    img = _imread_unicode(image_path)
    orig_img_shape = img.shape[:2]
    print(f"   原始图片尺寸: {orig_img_shape[1]} x {orig_img_shape[0]}")
    
    # 2. 创建 ONNX Runtime 会话并获取输入信息
    print("\n2. 加载 ONNX 模型...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    actual_provider = session.get_providers()[0]
    print(f"   使用的执行提供者: {actual_provider}")
    
    model_input_height, model_input_width, input_type, input_name = get_model_input_info(
        onnx_model_path, session
    )
    target_size = (model_input_height, model_input_width)
    print(f"   模型输入尺寸: {target_size[1]} x {target_size[0]}")
    print(f"   输入类型: {input_type}")
    
    # 3. 预处理图片
    print("\n3. 预处理图片...")
    if use_grayscale:
        print("   使用黑白图（仅第一通道有值）" if grayscale_ch0_only else "   使用黑白图（三通道同值）")
    img_tensor, prep_img_shape = preprocess_image(img, target_size, use_grayscale=use_grayscale, grayscale_ch0_only=grayscale_ch0_only)
    print(f"   预处理后尺寸: {img_tensor.shape}")
    
    # 4. 运行 ONNX 推理
    print("\n4. 运行 ONNX 推理...")
    prediction = run_onnx_inference(onnx_model_path, img_tensor, input_name)
    print(f"   预测输出形状: {prediction.shape}")
    
    # 5. 运行 Python NMS
    print("\n5. 运行 Python NMS...")
    print(f"   参数: conf_thres={conf_thres}, iou_thres={iou_thres}, max_det={max_det}, nc={nc}")
    detections = run_python_nms(prediction, conf_thres, iou_thres, max_det, nc)
    print(f"   检测到 {len(detections)} 个目标")
    
    # 6. 坐标转换
    if len(detections) > 0:
        print("\n6. 转换坐标到原始图片尺寸...")
        prep_img_shape_tuple = (target_size[0], target_size[1])  # (H, W)
        detections[:, :4] = scale_boxes(
            detections[:, :4], 
            prep_img_shape_tuple, 
            orig_img_shape
        )
        print(f"   坐标已转换: 从 {prep_img_shape_tuple} 到 {orig_img_shape}")
    
    # 7. 绘制检测结果
    print("\n7. 绘制检测结果...")
    img_for_draw = img
    if use_grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_for_draw = np.stack([gray, gray, gray], axis=-1)  # 三通道同值，便于绘制
    result_img = draw_detections(img_for_draw, detections)
    
    # 8. 保存结果
    if output_path is None:
        # 自动生成输出路径
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(onnx_model_path)
        output_path = os.path.join(output_dir, f"{base_name}_result.png")
    
    _imwrite_unicode(output_path, result_img)
    print(f"\n✅ 结果已保存到: {output_path}")
    print("=" * 60)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="使用 ONNX 模型和 Python NMS 进行推理")
    parser.add_argument('onnx_model', help='ONNX 模型路径')
    parser.add_argument('image', help='输入图片路径')
    parser.add_argument('-o', '--output', default=None, help='输出图片路径（默认：模型目录/图片名_result.png）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值（默认：0.25）')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU 阈值（默认：0.7）')
    parser.add_argument('--max-det', type=int, default=300, help='最大检测数量（默认：300）')
    parser.add_argument('--nc', type=int, default=2, help='类别数量（默认：2）')
    parser.add_argument('--grayscale', action='store_true', help='使用黑白图推理（三通道同值）')
    parser.add_argument('--grayscale-ch0-only', action='store_true', help='黑白图且仅第一通道有灰度值，第二、三通道为 0')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.onnx_model):
        print(f"❌ 错误: ONNX 模型文件不存在: {args.onnx_model}")
        return 1
    
    if not os.path.exists(args.image):
        print(f"❌ 错误: 图片文件不存在: {args.image}")
        return 1
    
    try:
        output_path = inference_with_python_nms(
            args.onnx_model,
            args.image,
            args.output,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
            nc=args.nc,
            use_grayscale=args.grayscale or args.grayscale_ch0_only,
            grayscale_ch0_only=args.grayscale_ch0_only
        )
        print(f"\n✅ 推理完成！输出文件: {output_path}")
        return 0
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
