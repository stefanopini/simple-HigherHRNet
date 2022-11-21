# YOLOv5 🚀 by Ultralytics, GPL-3.0 license - https://github.com/ultralytics/yolov5
"""
Export HigherHRNet in TensorRT inference engine format. Modified from yolov5 export.py function.
Usage:
    $ python export.py --weights pose_higher_hrnet_w32_512.pth --include engine

"""

import argparse
import os
import platform
import sys
import time
import warnings
sys.path.insert(1, os.getcwd())

from pathlib import Path

import onnx
import pandas as pd
import torch

from utils.yolov5.general import (LOGGER, Profile, check_requirements, check_version, colorstr, file_size,
                                  get_default_args, print_args, url2file)
from utils.yolov5.torch_utils import select_device, smart_inference_mode
from models.higherhrnet import HigherHRNet

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
MACOS = platform.system() == 'Darwin'  # macOS environment


def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TensorRT', 'engine', '.engine', False, True]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    check_requirements('onnx')

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0', 'output1']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx


@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=2, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt
    if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        export_onnx(model, im, file, 12, dynamic, simplify)
    else:  # TensorRT >= 8
        check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
    onnx = file.with_suffix('.onnx')

    LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 :
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return f, None


@smart_inference_mode()
def run(weights, hrnet_c, hrnet_j, imgsz=(512, 960), batch_size=1, device='cpu', include=('torchscript', 'onnx'),
        half=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=2):
    """Runs the model conversion from PyTorch to TensorRT.

    Args:
        weights (Path or str): HigherHRNet weights path
        hrnet_c (int): HigherHRNet number of channels
        hrnet_j (int): HigherHRNet number of joints
        imgsz (tuple(int, int)): image size (height, width)
        batch_size (int): batch size
        device (str): cuda device, i.e. 0 or 0,1,2,3 or cpu
        include (tuple(str, ...): include formats
        half (bool): FP16 half-precision export
        int8 (bool): CoreML/TF INT8 quantization (currently not supported)
        dynamic (bool): ONNX/TF/TensorRT: dynamic axes
        simplify (bool): ONNX: simplify model
        opset (int): ONNX: opset version (currently unused)
        verbose (bool): TensorRT: verbose log
        workspace (float): TensorRT: workspace size (GB)

    Returns:
        List of exported files/dirs
    """
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    engine = flags  # export booleans
    weights_file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)

    if half:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    if int8:
        raise NotImplementedError

    model = HigherHRNet(c=hrnet_c, nof_joints=hrnet_j)
    model.load_state_dict(torch.load(weights))
    model.cuda()

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    im = torch.zeros(batch_size, 3, imgsz[0],imgsz[1]).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for _ in range(2):
        y = model(im)  # dry runs

    # Exports
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if engine:  # TensorRT required before ONNX
        f[0], _ = export_engine(model, im, weights_file, half, dynamic, simplify, workspace, verbose)

    # f is a list of exported files/dirs
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)')

    return f


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/pose_higher_hrnet_w32_512.pth',
                        help='model.pth path(s)')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=32)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=list, default=[512, 960], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=1, help='TensorRT: workspace size (GB)')
    parser.add_argument('--include', nargs='+', default=['engine'], help='Export type to be included.')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
