# app/helpers/devices.py
import torch
import platform
import ctypes
from typing import List, Tuple, Dict, Optional
from pathlib import Path # Добавлен импорт Path

# Попытка импортировать tensorrt для проверки его доступности
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    pass # TensorRT не установлен
except Exception:
    pass # Другая ошибка при импорте


def get_available_devices() -> List[str]:
    """
    Возвращает список доступных вычислительных устройств в формате:
    "cpu", "cuda:0", "cuda:1", ..., "trt:0", "trt:1", ...
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append(f"cuda:{i}")
            if is_tensorrt_provider_available():
                devices.append(f"trt:{i}")
    return devices


def parse_device_setting(
    device_setting: str,
    trt_ep_options: Optional[Dict] = None
) -> Tuple[List, int, str]:
    """
    Разбирает строку настройки устройства (например, "cuda:0", "trt:1", "cpu")
    и возвращает список провайдеров ONNXRuntime, ID устройства CUDA и строку устройства PyTorch.

    Args:
        device_setting: Строка настройки устройства.
        trt_ep_options: Опциональный словарь с настройками для TensorRT Execution Provider.

    Returns:
        Tuple: (провайдеры, ID устройства CUDA, строка устройства PyTorch)
    """
    providers = []
    cuda_device_id = 0  # По умолчанию для CPU или если "cuda" без ID
    torch_device_string = "cpu"
    execution_provider_name = "CPUExecutionProvider" # Для удобства отслеживания

    if trt_ep_options is None:
        trt_ep_options = {} # Пустой словарь по умолчанию

    if device_setting.startswith("cuda"):
        parts = device_setting.split(':')
        if len(parts) > 1 and parts[1].isdigit():
            cuda_device_id = int(parts[1])
        torch_device_string = f"cuda:{cuda_device_id}"
        providers = [('CUDAExecutionProvider', {'device_id': str(cuda_device_id)})]
        execution_provider_name = "CUDAExecutionProvider"
    elif device_setting.startswith("trt:") and is_tensorrt_provider_available():
        parts = device_setting.split(':')
        if len(parts) > 1 and parts[1].isdigit():
            cuda_device_id = int(parts[1])
        else: # Если просто "trt", используем GPU 0 по умолчанию
            cuda_device_id = 0
        torch_device_string = f"cuda:{cuda_device_id}" # TensorRT работает поверх CUDA
        
        # Используем опции TensorRT из аргумента или создаем новые
        if not trt_ep_options:
            from app.processors.utils.engine_builder import create_tensorrt_provider_options
            trt_ep_options = create_tensorrt_provider_options(device_id=cuda_device_id)
        else:
            # Убедимся, что device_id установлен правильно
            trt_ep_options['device_id'] = cuda_device_id
            
        providers = [
            ('TensorrtExecutionProvider', trt_ep_options),
            ('CUDAExecutionProvider', {'device_id': str(cuda_device_id)}) # Фоллбэк, если TRT не справится
        ]
        execution_provider_name = "TensorrtExecutionProvider"
    elif device_setting == "cpu":
        providers = ['CPUExecutionProvider']
        torch_device_string = "cpu"
        execution_provider_name = "CPUExecutionProvider"
    else: # По умолчанию CUDA:0 или CPU, если CUDA недоступна
        if torch.cuda.is_available():
            torch_device_string = "cuda:0"
            providers = [('CUDAExecutionProvider', {'device_id': '0'})]
            execution_provider_name = "CUDAExecutionProvider"
        else:
            providers = ['CPUExecutionProvider']
            torch_device_string = "cpu"
            execution_provider_name = "CPUExecutionProvider"
            
    # Всегда добавляем CPUExecutionProvider в конец как фоллбэк, если он еще не там
    # и если это не единственный провайдер
    is_cpu_only_provider = (len(providers) == 1 and providers[0] == 'CPUExecutionProvider')
    
    # Проверяем, есть ли 'CPUExecutionProvider' в списке providers (как строка или первый элемент кортежа)
    cpu_provider_exists = any(
        p == 'CPUExecutionProvider' if isinstance(p, str) else p[0] == 'CPUExecutionProvider'
        for p in providers
    )

    if not is_cpu_only_provider and not cpu_provider_exists:
        providers.append('CPUExecutionProvider')
        
    return providers, cuda_device_id, torch_device_string


def get_onnx_device_type_and_id(torch_device_string: str, cuda_device_id: int) -> Tuple[str, int]:
    """
    Возвращает тип устройства ('cuda' или 'cpu') и ID устройства для ONNX Runtime IO Binding.
    """
    if torch_device_string.startswith("cuda"):
        return 'cuda', cuda_device_id
    return 'cpu', 0 # Для CPU device_id обычно 0


def get_tensorrt_plugin_path() -> Optional[str]:
    """
    Возвращает путь к библиотеке плагинов TensorRT в зависимости от платформы.
    """
    try:
        from app.processors.models_data import models_dir 
    except ImportError:
        print("Warning: models_dir not found, cannot determine TensorRT plugin path.")
        return None

    SYSTEM_PLATFORM = platform.system() # Получаем платформу здесь
    plugin_path_str = None

    if SYSTEM_PLATFORM == 'Windows':
        plugin_path_str = f'{models_dir}/grid_sample_3d_plugin.dll'
    elif SYSTEM_PLATFORM == 'Linux':
        plugin_path_str = f'{models_dir}/libgrid_sample_3d_plugin.so'
    else:
        return None
    
    plugin_path = Path(plugin_path_str)
    if plugin_path.exists():
        return str(plugin_path)
    else:
        # print(f"TensorRT plugin not found at: {plugin_path_str}")
        return None

def is_tensorrt_provider_available() -> bool:
    """
    Проверяет, доступен ли TensorRT провайдер в ONNX Runtime.
    
    Returns:
        bool: True, если TensorRT провайдер доступен, иначе False.
    """
    if not TENSORRT_AVAILABLE:
        return False
    
    try:
        import onnxruntime
        return 'TensorrtExecutionProvider' in onnxruntime.get_available_providers()
    except (ImportError, ModuleNotFoundError):
        return False