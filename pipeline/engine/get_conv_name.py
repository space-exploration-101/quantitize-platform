import re
import onnx
from typing import List, Dict, Set, Tuple
import os

def parse_model_structure_to_onnx_operators(model_file_path: str, layer_idx: int) -> List[str]:
    """
    解析YOLOv8模型结构文件，生成所有ONNX算子名称
    
    Args:
        model_file_path: 模型结构文件路径
        
    Returns:
        List[str]: 所有ONNX算子名称列表
    """
    with open(model_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    operators = []
    
    # 解析主干网络层 (0-7)
    operators, layer_type = parse_head_layers(content, layer_idx)
    
    return operators, layer_type



def parse_conv_layer(content: str, layer_idx: int) -> List[str]:
    """解析Conv层"""
    operators = []
    
    add_CBS(operators, layer_idx)
    return operators

def add_bottleneck(operators: List[str], layer_idx: int, bottleneck_index: int, extra_name: str = '', bool_add_conv: bool = True) -> List[str]:
    """解析Bottleneck模块"""
    add_CBS(operators, layer_idx,  extra_layer_name=f'{extra_name}/m.{bottleneck_index}/cv1')
    add_CBS(operators, layer_idx,  extra_layer_name=f'{extra_name}/m.{bottleneck_index}/cv2')
    if bool_add_conv:
        operators.append(f"/model.{layer_idx}{extra_name}/m.{bottleneck_index}/Add")
    return operators
def parse_c2f_layer(content: str, layer_idx: int) -> List[str]:
    """解析C2f层"""
    operators = []
    
    # cv1分支
    add_CBS(operators, layer_idx, extra_layer_name='/cv1')
    

    # 解析Bottleneck模块
    bottleneck_count = get_bottleneck_count(content, layer_idx)
    for i in range(bottleneck_count):
        add_bottleneck(operators, layer_idx, i)
        # cv2分支
    add_CBS(operators, layer_idx, extra_layer_name='/cv2')
    
    operators.append(f"/model.{layer_idx}/Concat")
    return operators
def add_CBS(operators: List[str], layer_idx: int, extra_layer_name: str = '') -> List[str]:
    """解析CBS层"""
    operators.append(f"/model.{layer_idx}{extra_layer_name}/conv/Conv")
    # operators.append(f"/model.{layer_idx}{extra_layer_name}/act/Sigmoid")
    # operators.append(f"/model.{layer_idx}{extra_layer_name}/act/Mul")
    return operators
def parse_c2_layer(content: str, layer_idx: int) -> List[str]:
    """解析C2层"""
    operators = []
    add_CBS(operators, layer_idx, '/cv1')
    add_CBS(operators, layer_idx, '/cv2')
    add_bottleneck(operators, layer_idx, 0,'/m', False)
    add_bottleneck(operators, layer_idx, 1,'/m', False)
    add_bottleneck(operators, layer_idx, 2,'/m', False)
   
    operators.append(f"/model.{layer_idx}/Concat")
    return operators

def get_bottleneck_count(content: str, layer_idx: int) -> int:
    """获取Bottleneck模块数量"""
    # 查找C2f层的Bottleneck数
    # 查找C2f层的Bottleneck数量
    # 方法1: 最宽松的匹配
    patterns = [
        # 宽松模式：允许任意字符和换行
        rf'\({layer_idx}\):\s*C2f.*?\((\d+)-(\d+)\):\s*(\d+)\s*x\s*Bottleneck',
        # 原始模式
        rf'\({layer_idx}\):\s*C2f\([^)]*\(m\):\s*ModuleList\([^)]*\((\d+)-(\d+)\):\s*(\d+)\s*x\s*Bottleneck',
        # 简化模式
        rf'\({layer_idx}\):\s*C2f\([^)]*\((\d+)-(\d+)\):\s*(\d+)\s*x\s*Bottleneck'
    ]
    
    for i, pattern in enumerate(patterns, 1):
        match = re.search(pattern, content, re.DOTALL)
        if match:
            count = int(match.group(3))
            print(f"层{layer_idx}: 模式{i}匹配成功，找到 {match.group(1)}-{match.group(2)} = {count} 个Bottleneck")
            return count
        else:
            print(f"层{layer_idx}: 模式{i}匹配失败")
    
    # 如果所有模式都失败，使用默认值
    print(f"层{layer_idx}: 所有模式都失败，使用默认值")
    default_counts = {2: 3, 4: 6, 6: 6, 8: 3}
    return default_counts.get(layer_idx, 0)
    

def parse_head_layers(content: str, layer_idx: int) -> List[str]:
    """解析检测头层"""
    operators = []
    # 查找该层的结构
    layer_pattern = rf'\({layer_idx}\):\s*(\w+)'
    layer_match = re.search(layer_pattern, content)
    print(layer_match,layer_idx)
    if layer_match:
        layer_type = layer_match.group(1)
        
        if layer_type == 'Conv':
            operators.extend(parse_conv_layer(content, layer_idx))
        elif layer_type == 'C2f':
            operators.extend(parse_c2f_layer(content, layer_idx))
        elif layer_type == 'C2':
            operators.extend(parse_c2_layer(content, layer_idx))
        elif layer_type == 'SPPF':
            operators.extend(parse_sppf_layer(content, layer_idx))
        elif layer_type == 'PoseDetect':
            operators.extend(parse_pose_detect_layer(content, layer_idx))
        elif layer_type == 'Upsample':
            operators.extend(parse_upsample_layer(content, layer_idx))
        elif layer_type == 'Concat':
            operators.extend(parse_concat_layer(content, layer_idx))
        elif layer_type == 'Pose':
            operators.extend(parse_pose_layer(content, layer_idx))
    return operators, layer_type

def parse_upsample_layer(content: str, layer_idx: int) -> List[str]:
    """解析Upsample层"""
    operators = []
    operators.append(f"/model.{layer_idx}/Resize")
    return operators
def parse_pose_layer(content: str, layer_idx: int) -> List[str]:
    """解析Pose层"""
    operators = []
    
    # cv2分支 - 4个尺度的检测分支
    for scale in range(4):  # 0-3个尺度
        # 第一个尺度：输入通道320
        add_CBS(operators, layer_idx,  f'/cv2.{scale}/cv2.{scale}.0')
        add_CBS(operators, layer_idx,  f'/cv2.{scale}/cv2.{scale}.1')
        add_conv(operators, layer_idx,  f'/cv2.{scale}/cv2.{scale}.2')
  
    # cv3分支 - 4个尺度的分类分支
    for scale in range(4):  # 0-3个尺度
        add_CBS(operators, layer_idx,  f'/cv3.{scale}/cv3.{scale}.0')
        add_CBS(operators, layer_idx,  f'/cv3.{scale}/cv3.{scale}.1')
        add_conv(operators, layer_idx,  f'/cv3.{scale}/cv3.{scale}.2')
        
    for scale in range(4):  # 0-3个尺度
        add_CBS(operators, layer_idx,  f'/cv4.{scale}/cv4.{scale}.0')
        add_CBS(operators, layer_idx,  f'/cv4.{scale}/cv4.{scale}.1')
        add_conv(operators, layer_idx,  f'/cv4.{scale}/cv4.{scale}.2')
    add_concat(operators, layer_idx,  f'')
    for i in range(1,8,1):
        add_concat(operators, layer_idx,  f'_{i}')
    # dfl分支
    operators.append(f"/model.{layer_idx}/dfl/conv/Conv")
    
    
    
    return operators
def add_concat(operators: List[str], layer_idx: int, extra_name: str = '') -> List[str]:
    """解析Concat层"""
    operators.append(f"/model.{layer_idx}/Concat{extra_name}")
    return operators
def add_conv(operators: List[str], layer_idx: int, extra_name: str = '') -> List[str]:
    """解析Conv层"""
    operators.append(f"/model.{layer_idx}{extra_name}/Conv")
    return operators
def parse_concat_layer(content: str, layer_idx: int) -> List[str]:
    """解析Concat层"""
    operators = []
    operators.append(f"/model.{layer_idx}/Concat")
    return operators

def parse_sppf_layer(content: str, layer_idx: int) -> List[str]:
    """解析SPPF层"""
    operators = []
    
    # cv1分支
    add_CBS(operators, layer_idx, '/cv1')
    
    # cv2分支
    add_CBS(operators, layer_idx,  '/cv2')
    
    # MaxPool层
    for i in range(3):
        if i == 0:
            operators.append(f"/model.{layer_idx}/m/MaxPool")
        else:
            operators.append(f"/model.{layer_idx}/m_{i}/MaxPool")
    
    return operators

def parse_pose_detect_layer(content: str, layer_idx: int) -> List[str]:
    """解析PoseDetect层"""
    operators = []
    
    # 检测分支
  
    add_conv(operators, layer_idx, '/cv2')
    # 分类分支
    add_conv(operators, layer_idx, '/cv3')
    
    # DFL分支 (只在最后一层)
    if layer_idx == 22:
        operators.append(f"/model.{layer_idx}/dfl/conv/Conv")
        
        # 姿态分支
        for i in range(4):  # 4个尺度
            add_conv(operators, layer_idx, f'/cv4/{i}/0')   
            add_conv(operators, layer_idx, f'/cv4/{i}/1')
            add_conv(operators, layer_idx, f'/cv4/{i}/2')
    
    return operators

def load_onnx_operators(onnx_file_path: str) -> Set[str]:
    """
    从ONNX文件中加载实际的算子名称
    
    Args:
        onnx_file_path: ONNX文件路径
        
    Returns:
        Set[str]: 实际ONNX算子名称集合
    """
    try:
        model = onnx.load(onnx_file_path)
        operators = list()
       
        for node in model.graph.node:
            operators.append( node.name)
        return [i for i in operators if not ('Sigmoid' in i and 'cv' in i)  and 'weight' not in i and 'Split' not in i  and 'Concat' not in i and 'Sigmoid' not in i]
    except Exception as e:
        print(f"加载ONNX文件失败: {e}")
        return list()

def verify_operators_correctness(model_file_path: str, onnx_file_path: str) -> Dict[str, any]:
    """
    验证算子名称的正确性
    
    Args:
        model_file_path: 模型结构文件路径
        onnx_file_path: ONNX文件路径
        
    Returns:
        Dict: 验证结果
    """
    print("=== 开始验证ONNX算子名称 ===")
    
    # 从模型结构文件生成算子名称
    print("1. 解析模型结构文件...")
    predicted_operators = set(parse_model_structure_to_onnx_operators(model_file_path))
    print(f"   预测的算子数量: {len(predicted_operators)}")
    
    # 从ONNX文件加载实际算子名称
    print("2. 加载ONNX文件...")
    actual_operators = load_onnx_operators(onnx_file_path)
    print(f"   实际的算子数量: {len(actual_operators)}")
    
    # 计算匹配情况
    actual_operators = set([c for c in actual_operators if 'Split' not in c])
    actual_operators = set([c for c in actual_operators if 'Slice' not in c])
    actual_operators = set([c for c in actual_operators if 'Reshape' not in c])
    actual_operators = set([c for c in actual_operators if 'Sub' not in c])
    actual_operators = set([c for c in actual_operators if 'Div' not in c])
    actual_operators = set([c for c in actual_operators if 'MaxPool' not in c])

    matched_operators = predicted_operators.intersection(actual_operators)
    missing_operators = predicted_operators - actual_operators
    extra_operators = actual_operators - predicted_operators
    
    # 计算准确率
    accuracy = len(matched_operators) / len(predicted_operators) * 100 if predicted_operators else 0
    recall = len(matched_operators) / len(actual_operators) * 100 if actual_operators else 0
    
    print(f"\n=== 验证结果 ===")
    print(f"匹配的算子数量: {len(matched_operators)}")
    print(f"缺失的算子数量: {len(missing_operators)}")
    print(f"额外的算子数量: {len(extra_operators)}")
    print(f"预测准确率: {accuracy:.2f}%")
    print(f"召回率: {recall:.2f}%")
    
    # 显示前10个匹配的算子
    print(f"\n=== 前10个匹配的算子 ===")
    for i, op in enumerate(sorted(matched_operators)[:10], 1):
        print(f"{i:2d}. {op}")
    
    # 显示前10个缺失的算子
    if missing_operators:
        print(f"\n=== 前10个缺失的算子 ===")
        for i, op in enumerate(sorted(missing_operators)[:10], 1):
            print(f"{i:2d}. {op}")
    
    # 显示前10个额外的算子
    if extra_operators:
        print(f"\n=== 前10个额外的算子 ===")
        for i, op in enumerate(sorted(extra_operators)[:100], 1):
            print(f"{i:2d}. {op}")
    
    return {
        'predicted_count': len(predicted_operators),
        'actual_count': len(actual_operators),
        'matched_count': len(matched_operators),
        'missing_count': len(missing_operators),
        'extra_count': len(extra_operators),
        'accuracy': accuracy,
        'recall': recall,
        'matched_operators': matched_operators,
        'missing_operators': missing_operators,
        'extra_operators': extra_operators
    }

def analyze_onnx_structure(onnx_file_path: str) -> Dict[str, List[str]]:
    """
    分析ONNX文件结构
    
    Args:
        onnx_file_path: ONNX文件路径
        
    Returns:
        Dict: ONNX结构分析结果
    """
    try:
        model = onnx.load(onnx_file_path)
        
        # 按类型分类算子
        operators_by_type = {}
        for node in model.graph.node:
            op_type = node.op_type
            if op_type not in operators_by_type:
                operators_by_type[op_type] = []
            operators_by_type[op_type].append(node.name)
        
        # 统计信息
        total_operators = sum(len(ops) for ops in operators_by_type.values())
        
        print(f"\n=== ONNX文件结构分析 ===")
        print(f"总算子数量: {total_operators}")
        print(f"算子类型数量: {len(operators_by_type)}")
        
        print(f"\n=== 各类型算子统计 ===")
        for op_type, ops in sorted(operators_by_type.items()):
            print(f"{op_type:15s}: {len(ops):3d} 个")
        
        
        return operators_by_type
        
    except Exception as e:
        print(f"分析ONNX文件失败: {e}")
        return {}

def get_all_onnx_operators(model_file_path: str) -> Dict[str, List[str]]:
    """
    获取所有ONNX算子名称，按类型分类
    
    Args:
        model_file_path: 模型结构文件路径
        
    Returns:
        Dict[str, List[str]]: 按类型分类的算子名称
    """
    operators = parse_model_structure_to_onnx_operators(model_file_path)
    
    # 按类型分类
    categorized = {
        'Conv': [],
        'Sigmoid': [],
        'Mul': [],
        'MaxPool': [],
        'All': operators
    }
    
    for op in operators:
        if '/conv/Conv' in op:
            categorized['Conv'].append(op)
        elif '/act/Sigmoid' in op:
            categorized['Sigmoid'].append(op)
        elif '/act/Mul' in op:
            categorized['Mul'].append(op)
        elif '/MaxPool' in op:
            categorized['MaxPool'].append(op)
    
    return categorized

def print_operators_summary(operators_dict: Dict[str, List[str]]):
    """打印算子摘要"""
    print("=== YOLOv8 Pose模型 ONNX算子摘要 ===")
    print(f"总算子数量: {len(operators_dict['All'])}")
    print(f"卷积层数量: {len(operators_dict['Conv'])}")
    print(f"Sigmoid激活数量: {len(operators_dict['Sigmoid'])}")
    print(f"Mul算子数量: {len(operators_dict['Mul'])}")
    print(f"MaxPool数量: {len(operators_dict['MaxPool'])}")
    
    print("\n=== 卷积层算子 ===")
    for i, conv in enumerate(operators_dict['Conv'][:10], 1):  # 显示前10个
        print(f"{i:2d}. {conv}")
    if len(operators_dict['Conv']) > 10:
        print(f"... 还有 {len(operators_dict['Conv']) - 10} 个卷积层")

# 主函数
def main():
    model_file_path = "/home/crab2/yolov8/Quantization-YOLOv8-main/model.txt"
    print("1. 解析模型结构文件...")
    # predicted_operators = parse_model_structure_to_onnx_operators(model_file_path, 2)
    onnx_file_path = "/home/crab2/yolov8/Quantization-YOLOv8-main/myquant2/temp_model2/wrs_fp16_final2_output.onnx"
    load_onnx_operators(onnx_file_path)
if __name__ == "__main__":
    main()