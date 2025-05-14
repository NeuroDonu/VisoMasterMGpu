# pylint: disable=no-member

import os
import sys
import logging
import platform
import ctypes
from pathlib import Path

try:
    import tensorrt as trt
except ModuleNotFoundError:
    pass

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

if 'trt' in globals():
    # Creazione di un'istanza globale di logger di TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.INFO) # pylint: disable=no-member
else:
    TRT_LOGGER = {}

# imported from https://github.com/warmshao/FasterLivePortrait/blob/master/scripts/onnx2trt.py
# adjusted to work with TensorRT 10.3.0
class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, custom_plugin_path=None, builder_optimization_level=3):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param custom_plugin_path: Path to the custom plugin library (DLL or SO).
        """
        if verbose:
            TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

        # Inizializza i plugin di TensorRT
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

        # Costruisce il builder di TensorRT e la configurazione usando lo stesso logger
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        # Imposta il limite di memoria del pool di lavoro a 3 GB
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 * (2 ** 30))  # 3 GB

        # Imposta il livello di ottimizzazione del builder (se fornito)
        self.config.builder_optimization_level = builder_optimization_level

        # Crea un profilo di ottimizzazione, se necessario
        profile = self.builder.create_optimization_profile()
        self.config.add_optimization_profile(profile)

        self.batch_size = None
        self.network = None
        self.parser = None

        # Carica plugin personalizzati se specificato
        if custom_plugin_path is not None:
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: %s", onnx_path)
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for net_input in inputs:
            self.batch_size = net_input.shape[0]
            log.info("Input '%s' with shape %s and dtype %s", net_input.name, net_input.shape, net_input.dtype)
        for net_output in outputs:
            log.info("Output %s' with shape %s and dtype %s", net_output.name, net_output.shape, net_output.dtype)

    def create_engine(self, engine_path, precision):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building %s Engine in %s", precision, engine_path)

        # Forza TensorRT a rispettare i vincoli di precisione
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        # Costruzione del motore serializzato
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)

        # Verifica che il motore sia stato serializzato correttamente
        if serialized_engine is None:
            raise RuntimeError("Errore nella costruzione del motore TensorRT!")

        # Scrittura del motore serializzato su disco
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: %s", engine_path)
            f.write(serialized_engine)

def change_extension(file_path, new_extension, version=None):
    """
    Change the extension of the file path and optionally prepend a version.
    """
    # Remove leading '.' from the new extension if present
    new_extension = new_extension.lstrip('.')

    # Create the new file path with the version before the extension, if provided
    if version:
        new_file_path = Path(file_path).with_suffix(f'.{version}.{new_extension}')
    else:
        new_file_path = Path(file_path).with_suffix(f'.{new_extension}')

    return str(new_file_path)

def onnx_to_trt(onnx_model_path, trt_model_path=None, precision="fp16", custom_plugin_path=None, verbose=False):
    # The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'"

    if trt_model_path is None:
        trt_version = trt.__version__
        trt_model_path = change_extension(onnx_model_path, "trt", version=trt_version)
    builder = EngineBuilder(verbose=verbose, custom_plugin_path=custom_plugin_path)

    builder.create_network(onnx_model_path)
    builder.create_engine(trt_model_path, precision)
    
def create_tensorrt_provider_options(trt_engine_path=None, device_id=0, fp16_enable=True):
    """
    Создает опции для TensorRT провайдера ONNX Runtime.
    
    Args:
        trt_engine_path (str, optional): Путь к файлу TensorRT движка или директории кэша.
        device_id (int): ID устройства CUDA.
        fp16_enable (bool): Включить ли поддержку FP16.
        
    Returns:
        dict: Словарь с опциями для TensorRT провайдера.
    """
    cache_path = os.path.dirname(trt_engine_path) if trt_engine_path else os.path.join(os.path.expanduser("~"), ".cache", "tensorrt")
    
    # Создаем директорию кэша, если она не существует
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    
    # Создаем опции для TensorRT провайдера
    provider_options = {
        'device_id': device_id,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_path,
        'trt_fp16_enable': fp16_enable,
        'trt_max_workspace_size': 3 * (2 ** 30),  # 3 GB
        'trt_engine_decryption_enable': False,
        'trt_engine_decryption_lib_path': '',
        'trt_dla_enable': False,
        'trt_dla_core': 0,
        'trt_dump_subgraphs': False,
        'trt_timing_cache_enable': True,
        'trt_force_sequential_engine_build': False,
        'trt_context_memory_sharing_enable': False,
        'trt_layer_norm_fp32_fallback': False,
        'trt_timing_cache_path': '',
        'trt_detailed_build_log': False,
        'trt_build_heuristics_enable': False,
        'trt_sparsity_enable': False,
        'trt_builder_optimization_level': 3,
        'trt_auxiliary_streams': 1,
        'trt_tactic_sources': '',
        'trt_extra_plugin_lib_paths': '',
        'trt_profile_min_shapes': '',
        'trt_profile_max_shapes': '',
        'trt_profile_opt_shapes': '',
        'trt_cuda_graph_enable': False,
    }
    
    return provider_options

class TensorRTProvider:
    """
    Класс для работы с TensorRT провайдером в ONNX Runtime.
    """
    
    @staticmethod
    def is_available():
        """
        Проверяет, доступен ли TensorRT.
        
        Returns:
            bool: True, если TensorRT доступен, иначе False.
        """
        try:
            import tensorrt
            import onnxruntime
            return 'TensorrtExecutionProvider' in onnxruntime.get_available_providers()
        except (ImportError, ModuleNotFoundError):
            return False
    
    @staticmethod
    def get_provider_and_options(device_id=0, trt_engine_path=None, fp16_enable=True):
        """
        Возвращает провайдер TensorRT и его опции для ONNX Runtime.
        
        Args:
            device_id (int): ID устройства CUDA.
            trt_engine_path (str, optional): Путь к файлу TensorRT движка или директории кэша.
            fp16_enable (bool): Включить ли поддержку FP16.
            
        Returns:
            tuple: (provider_name, provider_options) или (None, None), если TensorRT недоступен.
        """
        if not TensorRTProvider.is_available():
            log.warning("TensorRT provider is not available")
            return None, None
        
        provider_name = 'TensorrtExecutionProvider'
        provider_options = create_tensorrt_provider_options(
            trt_engine_path=trt_engine_path,
            device_id=device_id,
            fp16_enable=fp16_enable
        )
        
        return provider_name, provider_options
    
    @staticmethod
    def create_session(onnx_model_path, device_id=0, trt_engine_path=None, fp16_enable=True):
        """
        Создает сессию ONNX Runtime с TensorRT провайдером.
        
        Args:
            onnx_model_path (str): Путь к ONNX модели.
            device_id (int): ID устройства CUDA.
            trt_engine_path (str, optional): Путь к файлу TensorRT движка или директории кэша.
            fp16_enable (bool): Включить ли поддержку FP16.
            
        Returns:
            onnxruntime.InferenceSession: Сессия ONNX Runtime с TensorRT провайдером или None, если TensorRT недоступен.
        """
        import onnxruntime
        
        provider, options = TensorRTProvider.get_provider_and_options(
            device_id=device_id,
            trt_engine_path=trt_engine_path,
            fp16_enable=fp16_enable
        )
        
        if provider is None:
            log.warning("Using default CUDA provider instead of TensorRT")
            return onnxruntime.InferenceSession(
                onnx_model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        
        log.info(f"Creating ONNX Runtime session with TensorRT provider (device_id={device_id}, fp16={fp16_enable})")
        return onnxruntime.InferenceSession(
            onnx_model_path,
            providers=[
                (provider, options),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        )
