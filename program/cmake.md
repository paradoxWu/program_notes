# gcc/g++

1. 预处理 pre-processing //.i文件

> g++ -E sourcefile.cpp -o file.i

2. 编译 compiling /.s文件

> g++ -S  file.i  -o test.s

3. 汇编 assembling /.o文件

> g++ -c test.s -o test.o

4. 链接 linking //bin文件

> g++ test.o -o test

省略版本： `g++ sourcefile.cpp -o binfile`

## g++ 参数

-g 产生可被gnu调试器gdb调试的程序
： `g++ -g sourcefile.cpp -o binfile`
-O[n] 优化代码 n长设置0-3/常见 -O2
： `g++ -O2 sourcefile.cpp -o binfile`

-l 指定库文件（动态静态） 例如库文件叫glog
： `g++ -lglog sourcefile.cpp -o binfile`
-L 指定源文件
： `g++ -L/home/path/lib sourcefile.cpp -o binfile`
-I 指定头文件搜索路径
-std=c++11 设置编译标准
-D **定义宏** 很重要 cmake中常用

## 项目文件

include 头文件/对外接口
src 源文件/具体实现
install
deps
build
samples
test

### 库文件

常见命名：静态：libxxxx.a 动态 libxxxx.so
静态库 / 编译时用
汇编生成.o文件
归档生成静态库 ar rs 库名 .o文件
链接，生成可执行文件

> 1. g++ swap.cpp -c -I../include
> 2. ar rs libswap.a swap.o
> 3. g++ main.cpp -Iinclude -Lsrc -lswap -o staticmain
>    上述步骤就将swap的静态库链接上了main 生成一个可执行文件

动态库 /运行时用（运行时需要指定一下动态库路径）

> 1. g++ swap.cpp -I../include -fPIC -shared -o libswap.so
> 2. ar rs libswap.a swap.o
> 3. g++ main.cpp -Iinclude -Lsrc -lswap -o staticmain

依赖动态库的可执行文件被运行时需要指定路径

`LD_LIBRARY_PATH = ...... ./可执行文件`
而依赖静态库的直接运行即可

# CMAKE

参数使用括弧括起
参数之间使用空格或分号分开
指令是大小写无关的，参数和变量是大小写相关的

```cmake
set(HELLO hello.cpp)
add_executable(hello main.cpp hello.cpp)
ADD_EXECUTABLE(hello main.cpp ${HELLO})
```

变量使用${}方式取值，但是在 IF 控制语句中是直接使用变量名

## 重要指令

include_directories - 向工程添加多个特定的头文件搜索路径 --->相当于指定g++编译器的-I参数

```
# 将/usr/include/myincludefolder 和 ./include 添加到头文件搜索路径
include_directories(/usr/include/myincludefolder ./include)
```

link_directories - 向工程添加多个特定的库文件搜索路径 --->相当于指定g++编译器的-L参数
语法： link_directories(dir1 dir2 ...)

```
# 将/usr/lib/mylibfolder 和 ./lib 添加到库文件搜索路径
link_directories(/usr/lib/mylibfolder ./lib)
```

add_library - 生成库文件

```
# 通过变量 SRC 生成 libhello.so 共享库
add_library(hello SHARED ${SRC})
```

add_executable -生成可执行文件

```
# 编译main.cpp生成可执行文件main
add_executable(main main.cpp)
```

示例：以kestrel_crowdmax为例

这个编译方式是外部构建(out-of-source build)

```
mkdir build
cd build
cmake .. \
     -DKESTREL_DEVICE=CUDA11.0 \
     -DKESTREL_TEST=ON \
     -DCROWD_INTERNAL_TEST=ON \
     -DBUILD_SAMPLES=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DKESTREL_PERF=OFF \
     -DKESTREL_ASPECT=OFF \
     -DCMAKE_INSTALL_PREFIX="../install"\
     -DCROWD_DEBUG=OFF \
     -DACC_CHECK=OFF \
     -DWITH_CONCURRENCY=OFF
cmake --build . --target install -j8
cd ..

```

```cmake
#设置Cmake要求的最低版本 
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

#显式定义变量 设置版本与项目名称
SET(KESTREL_CROWD_OVERSEA_MAJOR_VERSION 1)
SET(KESTREL_CROWD_OVERSEA_MINOR_VERSION 4)
SET(KESTREL_CROWD_OVERSEA_PATCH_VERSION 0)
#版本 1.4.0
SET(KESTREL_CROWD_OVERSEA_VERSION ${KESTREL_CROWD_OVERSEA_MAJOR_VERSION}.${KESTREL_CROWD_OVERSEA_MINOR_VERSION}.${KESTREL_CROWD_OVERSEA_PATCH_VERSION})
#项目名称 kestrel_crowd_oversea 1.4.0
PROJECT(kestrel_crowd_oversea VERSION ${KESTREL_CROWD_OVERSEA_VERSION})

#${PROJECT_SOURCE_DIR}是指项目顶层目录路径
#${CMAKE_SOURCE_DIR} ${_SOURCE_DIR}这两个变量与其一致
INCLUDE(${PROJECT_SOURCE_DIR}/.keshub/keshub.cmake)
message("[CrowdMax] use cxx compiler version " ${CMAKE_CXX_COMPILER_VERSION})
IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-attributes -Wno-error=deprecated-declarations")
ELSEIF(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -Wno-attributes -Wno-error=deprecated-declarations")
endif()

#这与外部执行cmake命令是定义的变量有关 外部有这么一个外部参数-DCMAKE_BUILD_TYPE=Debug/Realease

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb -std=c++14 -Wno-error=deprecated-declarations")
ENDIF()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#设置默认参数 可在cmake执行命令行时设置
#例如 build_samples cmake .. -DBUILD_SAMPLES=OFF
OPTION(BUILD_SAMPLES  "build samples" ON)
OPTION(KESTREL_PERF   "enable the timer for performance" OFF)
OPTION(CROWD_DEBUG    "Output log to terminal for debugging" OFF)
OPTION(ACC_CHECK      "Output local log for accuracy check" OFF)
OPTION(KESTREL_TEST    "enable end to end unittest compilation" OFF)
OPTION(CROWD_INTERNAL_TEST    "enable internal unittest compilation" OFF)
OPTION(WITH_CONCURRENCY    "enable internal multi-thread optimization" OFF)

#根据cmake参数的设置 设置g++/gcc编译时的宏定义
IF(WITH_CONCURRENCY)
        ADD_DEFINITIONS(-DWITH_CONCURRENCY=1)
ELSEIF()
        ADD_DEFINITIONS(-DWITH_CONCURRENCY=0)
ENDIF()

IF(ACC_CHECK)
        ADD_DEFINITIONS(-DACC_CHECK=1)
ENDIF()

IF(CROWD_DEBUG)
        ADD_DEFINITIONS(-DCROWD_DEBUG=1)
ENDIF()

IF (KESTREL_PERF)
    ADD_DEFINITIONS(-DKESTREL_PERF)
    IF (KESTREL_ASPECT)
        ADD_DEFINITIONS(-DKESTREL_ASPECT)
    ENDIF()
ENDIF()

IF (KESTREL_DEVICE MATCHES "^CUDA")
    SET(KESTREL_CUDA ON)
    FIND_PACKAGE(CUDA REQUIRED)
    ADD_DEFINITIONS(-DKESTREL_CUDA=1)
ELSEIF (KESTREL_DEVICE MATCHES "^Atlas")
    SET(KESTREL_ATLAS ON)
    ADD_DEFINITIONS(-DKESTREL_ATLAS=1)
ELSEIF (KESTREL_DEVICE MATCHES "^35")
    SET(KESTREL_HISI ON)
    ADD_DEFINITIONS(-DKESTREL_HISI=1)
ELSEIF (KESTREL_DEVICE MATCHES "^S100")
    SET(KESTREL_STPU ON)
    ADD_DEFINITIONS(-DKESTREL_STPU=1)
    IF(KESTREL_DEVICE MATCHES "S100SA")
            ADD_DEFINITIONS(-DKESTREL_S100SA=1)
    ENDIF()
    IF(KESTREL_DEVICE MATCHES "S100ACC")
            ADD_DEFINITIONS(-DKESTREL_S100ACC=1)
    ENDIF()
ENDIF()

IF(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*")
        ADD_DEFINITIONS(-mfloat-abi=softfp -mfpu=neon)
ENDIF()
ADD_DEFINITIONS(-DKESTREL_LOG_LABEL="CrowdMax")

#指定头文件搜索路径 相当于g++ 的-I参数
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
#指定库文件搜索路径 相当于g++ 的-L参数
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/deps/libs)

#在指定路径下查找所有符合正则表达式的文件，这里将所有源文件赋值给SRC变量
#下面可以用${SRC} 代指所有规定路径下*.cpp文件
FILE(GLOB_RECURSE SRC src/crowd_oversea/*.cpp src/*_c.cpp)

#kestrel自己定义的命令，类似于add_library+target_link_libraies(),为生成库文件且为目标增加需要链接的库
#这里是为SRC链接上下方的这些kestrel_aux等库
KESTREL_LIBRARY(kestrel_crowd_oversea
        SRC ${SRC}
        LINKS
        kestrel_aux
        kestrel
        kestrel_core
        PPL3CV_static
        PPL3Core_static
        )
KESTREL_LIBRARY_INSTALL(kestrel_crowd_oversea ${KESTREL_INSTALL_LIB_DIR})
KESTREL_INSTALL_DEPS(TARGETS kestrel INCLUDE SHARED_LIB)
KESTREL_INSTALL_DEPS(TARGETS kestrel_ppl kestrel_nart demuxer colo hunter harpy hermes relation  SHARED_LIB)

IF(KESTREL_CUDA)
    KESTREL_INSTALL_DEPS(TARGETS
        kestrel_mixnet
        kestrel_caffe
        decoder
        SHARED_LIB OPTIONAL)
ELSEIF(KESTREL_HISI)
    KESTREL_INSTALL_DEPS(TARGETS
        hidecoder
        hidevice
        hisvp
        SHARED_LIB OPTIONAL)
ELSEIF(KESTREL_ATLAS)
    KESTREL_INSTALL_DEPS(TARGETS
        acldecoder
        acldevice
        SHARED_LIB OPTIONAL)
ELSEIF(KESTREL_STPU)
    KESTREL_INSTALL_DEPS(TARGETS
        demuxer
        stpu_decoder
        stpu_encoder
        stpu_device
        SHARED_LIB OPTIONAL)
ENDIF()

#执行子目录下的cmakelist
IF(BUILD_SAMPLES)
    ADD_SUBDIRECTORY(samples)
ENDIF()

IF(KESTREL_TEST)
  ADD_SUBDIRECTORY(test)
ENDIF()

IF(CROWD_INTERNAL_TEST)
        ADD_DEFINITIONS(-DGTEST)
  ADD_SUBDIRECTORY(src/gtest)
ENDIF()

#install project and lib
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${KESTREL_INSTALL_INC_DIR})
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/samples/ DESTINATION samples USE_SOURCE_PERMISSIONS)
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/python/ DESTINATION python USE_SOURCE_PERMISSIONS)
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/doc/ DESTINATION doc )
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/schema/ DESTINATION schema )

FILE(GLOB_RECURSE MODEL_SCRIPTS ${PROJECT_SOURCE_DIR}/models/*.sh ${PROJECT_SOURCE_DIR}/models/*.yml)
INSTALL(FILES ${MODEL_SCRIPTS} DESTINATION samples )
INSTALL(FILES README.md DESTINATION .)
INSTALL(FILES .signignore DESTINATION .)

```
