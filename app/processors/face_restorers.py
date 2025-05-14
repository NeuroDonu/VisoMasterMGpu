
from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans

from app.helpers.devices import get_onnx_device_type_and_id

if TYPE_CHECKING:
    from app.processors.models_processor_v2 import ModelsProcessorV2

class FaceRestorers:
    def __init__(self, models_processor: 'ModelsProcessorV2'):
        self.models_processor = models_processor
        
    def _get_device_info(self):
        """
        Получает информацию об устройстве для ONNX Runtime IO Binding.
        
        Returns:
            tuple: (device_type, device_id)
        """
        return get_onnx_device_type_and_id(
            self.models_processor.torch_device_string, 
            self.models_processor.cuda_device_id
        )
            
    def _sync_device(self):
        """
        Синхронизирует устройство после выполнения операций.
        """
        if self.models_processor.device == "cuda" or self.models_processor.device.startswith("cuda:"):
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()

    def apply_facerestorer(self, swapped_face_upscaled, restorer_det_type, restorer_type, restorer_blend, fidelity_weight, detect_score):
        temp = swapped_face_upscaled
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)
        t1024 = v2.Resize((1024, 1024), antialias=False)
        t2048 = v2.Resize((2048, 2048), antialias=False)

        # If using a separate detection mode
        if restorer_det_type == 'Blend' or restorer_det_type == 'Reference':
            if restorer_det_type == 'Blend':
                # Set up Transformation
                dst = self.models_processor.arcface_dst * 4.0
                dst[:,0] += 32.0

            elif restorer_det_type == 'Reference':
                try:
                    dst, _, _ = self.models_processor.run_detect_landmark(swapped_face_upscaled, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='5', score=detect_score/100.0, from_points=False)
                except Exception as e: # pylint: disable=broad-except
                    print(f"exception: {e}")
                    return swapped_face_upscaled

            # Return non-enhanced face if keypoints are empty
            if not isinstance(dst, np.ndarray) or len(dst)==0:
                return swapped_face_upscaled
            
            tform = trans.SimilarityTransform()
            try:
                tform.estimate(dst, self.models_processor.FFHQ_kps)
            except:
                return swapped_face_upscaled
            # Transform, scale, and normalize
            temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            temp = v2.functional.crop(temp, 0,0, 512, 512)

        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

        if restorer_type == 'GPEN-256':
            temp = t256(temp)

        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()

        if restorer_type == 'GFPGAN-v1.4':
            self.run_GFPGAN(temp, outpred)

        elif restorer_type == 'CodeFormer':
            self.run_codeformer(temp, outpred, fidelity_weight)

        elif restorer_type == 'GPEN-256':
            outpred = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
            self.run_GPEN_256(temp, outpred)

        elif restorer_type == 'GPEN-512':
            self.run_GPEN_512(temp, outpred)

        elif restorer_type == 'GPEN-1024':
            temp = t1024(temp)
            outpred = torch.empty((1, 3, 1024, 1024), dtype=torch.float32, device=self.models_processor.device).contiguous()
            self.run_GPEN_1024(temp, outpred)

        elif restorer_type == 'GPEN-2048':
            temp = t2048(temp)
            outpred = torch.empty((1, 3, 2048, 2048), dtype=torch.float32, device=self.models_processor.device).contiguous()
            self.run_GPEN_2048(temp, outpred)

        elif restorer_type == 'RestoreFormer++':
            self.run_RestoreFormerPlusPlus(temp, outpred)

        elif restorer_type == 'VQFR-v2':
            self.run_VQFR_v2(temp, outpred, fidelity_weight)

        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)

        if restorer_type == 'GPEN-256' or restorer_type == 'GPEN-1024' or restorer_type == 'GPEN-2048':
            outpred = t512(outpred)

        # Invert Transform
        if restorer_det_type == 'Blend' or restorer_det_type == 'Reference':
            outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(restorer_blend)/100.0
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred

    def run_GFPGAN(self, image, output):
        if not self.models_processor.models['GFPGANv1.4']:
            self.models_processor.models['GFPGANv1.4'] = self.models_processor.load_model('GFPGANv1.4')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['GFPGANv1.4'].io_binding()
        io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['GFPGANv1.4'].run_with_iobinding(io_binding)

    def run_GPEN_256(self, image, output):
        if not self.models_processor.models['GPENBFR256']:
            self.models_processor.models['GPENBFR256'] = self.models_processor.load_model('GPENBFR256')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['GPENBFR256'].io_binding()
        io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['GPENBFR256'].run_with_iobinding(io_binding)

    def run_GPEN_512(self, image, output):
        if not self.models_processor.models['GPENBFR512']:
            self.models_processor.models['GPENBFR512'] = self.models_processor.load_model('GPENBFR512')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['GPENBFR512'].io_binding()
        io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['GPENBFR512'].run_with_iobinding(io_binding)

    def run_GPEN_1024(self, image, output):
        if not self.models_processor.models['GPENBFR1024']:
            self.models_processor.models['GPENBFR1024'] = self.models_processor.load_model('GPENBFR1024')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['GPENBFR1024'].io_binding()
        io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,1024,1024), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['GPENBFR1024'].run_with_iobinding(io_binding)

    def run_GPEN_2048(self, image, output):
        if not self.models_processor.models['GPENBFR2048']:
            self.models_processor.models['GPENBFR2048'] = self.models_processor.load_model('GPENBFR2048')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['GPENBFR2048'].io_binding()
        io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,2048,2048), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['GPENBFR2048'].run_with_iobinding(io_binding)

    def run_codeformer(self, image, output, fidelity_weight_value=0.9):
        if not self.models_processor.models['CodeFormer']:
            self.models_processor.models['CodeFormer'] = self.models_processor.load_model('CodeFormer')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['CodeFormer'].io_binding()
        io_binding.bind_input(name='x', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        w = np.array([fidelity_weight_value], dtype=np.double)
        io_binding.bind_cpu_input('w', w)
        io_binding.bind_output(name='y', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['CodeFormer'].run_with_iobinding(io_binding)

    def run_VQFR_v2(self, image, output, fidelity_ratio_value):
        if not self.models_processor.models['VQFRv2']:
            self.models_processor.models['VQFRv2'] = self.models_processor.load_model('VQFRv2')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        assert fidelity_ratio_value >= 0.0 and fidelity_ratio_value <= 1.0, 'fidelity_ratio must in range[0,1]'
        fidelity_ratio = torch.tensor(fidelity_ratio_value).to(self.models_processor.device)

        io_binding = self.models_processor.models['VQFRv2'].io_binding()
        io_binding.bind_input(name='x_lq', device_type=device_type, device_id=device_id, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_input(name='fidelity_ratio', device_type=device_type, device_id=device_id, element_type=np.float32, shape=fidelity_ratio.size(), buffer_ptr=fidelity_ratio.data_ptr())
        io_binding.bind_output('enc_feat', device_type, device_id)
        io_binding.bind_output('quant_logit', device_type, device_id)
        io_binding.bind_output('texture_dec', device_type, device_id)
        io_binding.bind_output(name='main_dec', device_type=device_type, device_id=device_id, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=output.data_ptr())

        self._sync_device()
        self.models_processor.models['VQFRv2'].run_with_iobinding(io_binding)

    def run_RestoreFormerPlusPlus(self, image, output):
        if not self.models_processor.models['RestoreFormerPlusPlus']:
            self.models_processor.models['RestoreFormerPlusPlus'] = self.models_processor.load_model('RestoreFormerPlusPlus')

        # Получаем правильный device_type и device_id для ONNX Runtime
        device_type, device_id = self._get_device_info()

        io_binding = self.models_processor.models['RestoreFormerPlusPlus'].io_binding()
        io_binding.bind_input(name='input', device_type=device_type, device_id=device_id, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='2359', device_type=device_type, device_id=device_id, element_type=np.float32, shape=output.size(), buffer_ptr=output.data_ptr())
        io_binding.bind_output('1228', device_type, device_id)
        io_binding.bind_output('1238', device_type, device_id)
        io_binding.bind_output('onnx::MatMul_1198', device_type, device_id)
        io_binding.bind_output('onnx::Shape_1184', device_type, device_id)
        io_binding.bind_output('onnx::ArgMin_1182', device_type, device_id)
        io_binding.bind_output('input.1', device_type, device_id)
        io_binding.bind_output('x', device_type, device_id)
        io_binding.bind_output('x.3', device_type, device_id)
        io_binding.bind_output('x.7', device_type, device_id)
        io_binding.bind_output('x.11', device_type, device_id)
        io_binding.bind_output('x.15', device_type, device_id)
        io_binding.bind_output('input.252', device_type, device_id)
        io_binding.bind_output('input.280', device_type, device_id)
        io_binding.bind_output('input.288', device_type, device_id)

        self._sync_device()
        self.models_processor.models['RestoreFormerPlusPlus'].run_with_iobinding(io_binding)