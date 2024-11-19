TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += qt

DISTFILES += \
    param_mine.ini

INCLUDEPATH += E:/opencv/build/build_debug/install/include

LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_core451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_highgui451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_imgproc451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_features2d451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_calib3d451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_video451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_videoio451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_imgcodecs451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_dnn451d
LIBS += -L"E:/opencv/build/build_debug/install/x64/vc16/lib" -lopencv_tracking451d


DESTDIR = ./debug
CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda

# This makes the .cu files appear in your project
# OTHER_FILES += \
#     vectorAdd.cu
CUDA_SOURCES += \
    main.cu

#-------------------------------------------------

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# CUDA settings
#CUDA_DIR = $$(CUDA_PATH)
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/"
SYSTEM_NAME = x64                   # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
CUDA_ARCH = sm_50                   # Type of CUDA architecture
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include \


# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \


# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Add the necessary libraries
CUDA_LIB_NAMES = cudart_static kernel32 user32 gdi32 winspool comdlg32 \
                 advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32 \
                 #freeglut glew32

for(lib, CUDA_LIB_NAMES) {
    CUDA_LIBS += -l$$lib
}
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    --compile -cudart static -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
