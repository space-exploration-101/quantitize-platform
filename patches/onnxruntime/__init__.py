"""
本地 onnxruntime 模块

这是一个部分实现的 onnxruntime 模块，主要用于提供自定义的量化工具。
对于完整的 onnxruntime 功能，会回退到系统安装的版本。
"""

import sys
import os

# 获取当前模块的目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# 缓存系统版本的 onnxruntime
_system_ort = None

def _get_system_ort():
    """获取系统版本的 onnxruntime"""
    global _system_ort
    if _system_ort is None:
        # 临时移除本地路径，避免递归导入
        _was_in_path = False
        if _parent_dir in sys.path:
            sys.path.remove(_parent_dir)
            _was_in_path = True
        
        try:
            # 清除 sys.modules 中的本地版本
            if 'onnxruntime' in sys.modules:
                mod = sys.modules['onnxruntime']
                if hasattr(mod, '__file__') and mod.__file__ and ('/patches/onnxruntime' in mod.__file__.replace('\\', '/') or mod.__file__.endswith('patches/onnxruntime/__init__.py')):
                    del sys.modules['onnxruntime']
            
            import onnxruntime
            _system_ort = onnxruntime
        finally:
            # 恢复路径
            if _was_in_path:
                sys.path.insert(0, _parent_dir)
    return _system_ort

# 使用 __getattr__ 来代理所有未定义的属性到系统版本
def __getattr__(name):
    """代理未定义的属性到系统版本的 onnxruntime"""
    if name == 'quantization':
        # 延迟导入本地 quantization 模块
        try:
            from . import quantization
            return quantization
        except ImportError:
            pass
    
    # 从系统版本获取属性
    system_ort = _get_system_ort()
    if system_ort is not None:
        return getattr(system_ort, name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 预先导入系统版本的关键属性，避免每次都调用 __getattr__
try:
    _system_ort = _get_system_ort()
    if _system_ort is not None:
        # 将系统版本的关键类直接设置到模块命名空间
        for attr_name in ['GraphOptimizationLevel', 'InferenceSession', 'SessionOptions']:
            if hasattr(_system_ort, attr_name):
                setattr(sys.modules[__name__], attr_name, getattr(_system_ort, attr_name))
except Exception:
    pass

__all__ = ['quantization']
