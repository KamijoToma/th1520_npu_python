SHL_PATH != python -m shl --whereis th1520
EXAMPLE_FILES != ls example
SHL_INCLUDE=$(SHL_PATH)/include
SHL_LIB=$(SHL_PATH)/lib
TH1520_LIB=libshl_th1520.so.2

CC = gcc
CFLAGS = -O0 -g -mabi=lp64d -fPIC
INCLUDES = \
    -I$(SHL_INCLUDE)/shl_public/
LDFLAGS = -L$(SHL_LIB) \
	-lshl
#ctypesgen -I th1520/include -llibshl_th1520.so.2 -llibmodel.so th1520/include/*.h th1520/include/shl_public/*.h th1520/include/csinn/*.h -o shl.py

shl_lib_th1520: libmodel.so model.h
ifeq ($(SHL_PATH),)
	$(error "Failed to detect SHL_PATH, check if shl-python lib is installed")
endif
	$(info "SHL_PATH: " $(SHL_PATH))
	ctypesgen -I $(SHL_INCLUDE) -l$(TH1520_LIB) -llibmodel.so $(SHL_INCLUDE)/*.h $(SHL_INCLUDE)/shl_public/*.h $(SHL_INCLUDE)/csinn/*.h model.h -o shlbind.py

model.o: model.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

libmodel.so: model.o
	$(CC) $< -o $@ -shared -fPIC $(LDFLAGS) 

clean:
	rm -f *.o *.so shlbind.py $(EXAMPLE_FILES) *.whl *.bm

prepare_test:
	cp example/* .

do_test: prepare_test shl_lib_th1520
	python shllib.py

test: do_test clean
	@echo "Enjoy!"

numpy-1.25.0-cp311-cp311-linux_riscv64.whl:
	curl -OL "https://github.com/zhangwm-pt/prebuilt_whl/raw/refs/heads/python3.11/numpy-1.25.0-cp311-cp311-linux_riscv64.whl"

opencv_python-4.5.4+4cd224d-cp311-cp311-linux_riscv64.whl:
	curl -OL "https://github.com/zhangwm-pt/prebuilt_whl/raw/refs/heads/python3.11/opencv_python-4.5.4+4cd224d-cp311-cp311-linux_riscv64.whl"

download_requirements: numpy-1.25.0-cp311-cp311-linux_riscv64.whl opencv_python-4.5.4+4cd224d-cp311-cp311-linux_riscv64.whl
	

install_requirements: download_requirements clean
	pip install ./numpy-1.25.0-cp311-cp311-linux_riscv64.whl
	pip install ./opencv_python-4.5.4+4cd224d-cp311-cp311-linux_riscv64.whl
	pip install ctypesgen shl-python