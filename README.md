# TH1520 NPU Python API

PoC developed and verified on LPi4A.

This repo implements a python api to integrate TH1520 NPU inference with popular Python deep-learning infra NumPy.

The `example` folder shows a example using NPU to inference yolov5n with minimized dependence to C code.

## Have a try

1. Clone this repo to TH1520 devices like LPi4A
2. Install `python3-opencv` package using apt and create a python virtual environment with argument `--system-site-packages` and activate it
3. Install requirements by run `make install_requirements`. To generate python bindings, you will need `ctypesgen` and 
`shl-python`; To run examples, you need `numpy` `opencv-python`, which should be installed via system package manager because **prebuilt opencv package lacks of video support**. If you do not care, you can download prebuild wheels from [Here](https://github.com/zhangwm-pt/prebuilt_whl)
4. Run example using the commands shown below. Be sure to load npu driver by `sudo modprobe vha` and set correct device permission by `sudo chmod a+rw /dev/vha*`


| Name | Description | Command |
| ---- | ----------- | ------- |
| yolo_singleimage | A basic yolov5n example shows the usage of shllib.py (~20ms inference time) | `make test_yolo_single` |
| yolo_video | Shows object detection using video (~5fps) | `make test_yolo_video` |
| yolo_video_para | Advanced example using multiprocess to parallize yolov5n video detection (~7fps) | `make test_yolo_video_para` |

## Run your own model

1. Use x86 hhb docker image to generate `hhb_out` folder. See documentation and example at [here](https://www.yuque.com/za4k4z/yp3bry/gd20dgcs37dycevo)
2. Copy `hhb_out/model.c` and `hhb_out/model.params` to this repo
3. Take a look at `shllib.py`, write necessary code to preprocess and postprocess data, as `Csinn` class only implements core inference process, it accepts `list[np.ndarray]` as input tensors and output the inferenced result as `list[np.ndarray]`
4. Generate required native library and python bindings by run `make shl_lib_th1520`
5. Run your code

## Bugs & Todos

1. Fix possible memory-leaks
2. Support CPU inference
3. Support usage of JIT compiled `shl.hhb.bm` file

## Optimizations

### Sigmoid() optimize

replaced sigmoid function with `x/(2+2*abs(x))+0.5`, reduced sigmoid() cost from 44.54s to 16.4s

## License

Apache 2.0
