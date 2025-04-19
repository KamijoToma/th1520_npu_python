# TH1520 NPU Python API

PoC developed and verified on LPi4A.

This repo implements a python api to integrate TH1520 NPU inference with popular Python deep-learning infra NumPy.

The `example` folder shows a example using NPU to inference yolov5n with minimized dependence to C code.

## Have a try

1. Clone this repo to TH1520 devices like LPi4A
2. Create a python virtual environment and activate it
3. Install requirements by run `make install_requirements`. To generate python bindings, you will need `ctypesgen` and 
`shl-python`; To run examples, you need `numpy` `opencv-python`. You can download prebuild wheels from [Here](https://github.com/zhangwm-pt/prebuilt_whl)
4. Run example by `make test`. Be sure to load npu driver by `modprobe vha` and set correct device permission by `sudo chmod a+rw /dev/vha*`

## Run your own model

1. Use x86 hhb docker image to generate `hhb_out` folder. See documentation and example at [here](https://www.yuque.com/za4k4z/yp3bry/gd20dgcs37dycevo)
2. Copy `hhb_out/model.c` to this repo
3. Take a look at `shllib.py`, write necessary code to preprocess and postprocess data, as `Csinn` class only implements core inference process, it accepts `list[np.ndarray]` as input tensors and output the inferenced result as `list[np.ndarray]`
4. Generate required native library and python bindings by run `make shl_lib_th1520`
5. Run your code

## Bugs & Todos

1. Fix possible memory-leaks
2. Support CPU inference

## License

Apache 2.0
