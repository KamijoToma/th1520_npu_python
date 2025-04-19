SHL_PATH != python -m shl --whereis th1520
EXAMPLE_FILES != find example/ -type f -printf "\"%f\" "
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

prepare_test_yolo_single:
	cp example/yolo_singleimage/* .

prepare_test_yolo_video: example/yolo_video/input.mp4
	cp example/yolo_video/* .
	mkdir -p output

do_test_yolo_single: prepare_test_yolo_single shl_lib_th1520
	python shllib.py

do_test_yolo_video_para: prepare_test_yolo_video shl_lib_th1520
	python yolov5n_para.py
	@echo "check output dir for processed images!"

do_test_yolo_video: prepare_test_yolo_video shl_lib_th1520
	python yolov5n_video.py
	@echo "check output dir for processed images!"

test_%: do_test_% clean
	@echo "Enjoy!"

install_requirements:
	pip install ctypesgen shl-python
