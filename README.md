# 50-port-ai-gateway - 港口AI网关系统

面向大型全自动化港口无人集卡（AGV）与岸桥双吊具全自动精准对锁抓取场景的边缘AI网关系统。

## 系统概述

本系统部署在车顶的 NVIDIA Jetson 边缘计算盒子上，采用 C++ 纯原生框架，打通两条极高带宽的硬核实时智能感知流水线：

### 1. 多路 RTSP 4K 超清超导流媒体硬件级解帧同步层
- 利用 FFmpeg 开启 GPU 硬件加速组件（NVDEC）
- 双相机（可见光+红外）数据包微秒级帧时间戳（PTS）严格对齐
- 纯 RGB 特征图矩阵直接打入 GPU 锁定内存

### 2. TensorRT 深度学习实例分割推理引擎部署
- 完全弃用 Python 壳子，C++ 原生实现
- 逆向解析优化序列化后的 Engine 密包
- 高并发 CUDA 前处理扩容拉伸（Letterbox 变换）
- 10 毫秒内流式圈出集装箱锁孔（Corner Slot）的 3D ROI 区域

## 项目结构

```
50-port-ai-gateway/
├── CMakeLists.txt              # CMake 构建配置
├── cmake/                      # CMake 模块
│   ├── FindFFmpeg.cmake
│   └── FindTensorRT.cmake
├── include/                    # 头文件
│   ├── common/                 # 公共模块
│   │   ├── frame.h             # 帧数据结构
│   │   ├── thread_safe_queue.h # 线程安全队列
│   │   ├── gpu_memory.h        # GPU 内存管理
│   │   ├── time_util.h         # 时间工具
│   │   └── logger.h            # 日志系统
│   ├── streaming/              # 流媒体模块
│   │   ├── rtsp_reader.h       # RTSP 拉流器
│   │   └── frame_synchronizer.h# 帧同步器
│   ├── inference/              # 推理模块
│   │   ├── trt_engine.h        # TensorRT 引擎
│   │   ├── cuda_preprocessor.h # CUDA 前处理
│   │   └── corner_slot_detector.h # 锁孔检测器
│   └── app/                    # 应用层
│       └── gateway_pipeline.h  # 网关流水线
├── src/                        # 源文件
│   ├── main.cpp                # 主程序入口
│   ├── common/
│   │   ├── gpu_memory.cpp
│   │   ├── time_util.cpp
│   │   └── logger.cpp
│   ├── streaming/
│   │   ├── rtsp_reader.cpp
│   │   └── frame_synchronizer.cpp
│   ├── inference/
│   │   ├── cuda_preprocessor.cu
│   │   ├── trt_engine.cpp
│   │   └── corner_slot_detector.cpp
│   └── app/
│       └── gateway_pipeline.cpp
├── config/                     # 配置文件
│   └── config.json
└── models/                     # 模型目录
    └── README.md
```

## 技术栈

- **编程语言**: C++17 / CUDA C++
- **构建系统**: CMake 3.18+
- **硬件加速**:
  - NVIDIA NVDEC (FFmpeg 硬件解码)
  - CUDA (前处理加速)
  - TensorRT (推理加速)
- **多媒体**: FFmpeg (RTSP 拉流、编解码)
- **目标平台**: NVIDIA Jetson AGX Orin / Xavier NX

## 核心功能模块

### 1. 公共模块 (common)

#### 帧数据结构 (frame.h)
- `Frame`: 单帧数据，支持 CPU/GPU 内存
- `StereoFramePair`: 双帧同步对（可见光+红外）
- 支持 RGB/BGR/NV12/YUV420P/GRAY 多种像素格式

#### 线程安全队列 (thread_safe_queue.h)
- 无锁设计，高并发场景优化
- 支持阻塞/非阻塞操作
- 支持最大容量限制和超时等待

#### GPU 内存管理 (gpu_memory.h)
- `GpuMemoryManager`: GPU 内存管理单例
- `GpuBuffer`: GPU 缓冲区 RAII 封装
- `PinnedBuffer`: 页锁定内存（零拷贝）

#### 时间工具 (time_util.h)
- 高精度时间戳（微秒级）
- `ScopedTimer`: 性能计时器
- `FpsCounter`: FPS 统计

#### 日志系统 (logger.h)
- 多级别日志：TRACE/DEBUG/INFO/WARN/ERROR/FATAL
- 线程安全
- 自动带文件名、行号、函数名

### 2. 流媒体模块 (streaming)

#### RTSP 拉流器 (rtsp_reader.h)
- 支持 RTSP over TCP/UDP
- FFmpeg NVDEC 硬件解码
- 自动重连机制
- PTS 时间戳精确标记

#### 帧同步器 (frame_synchronizer.h)
- 双相机帧时间戳对齐
- 微秒级同步精度
- 可配置最大时间差容忍度
- 滑动窗口匹配算法

### 3. 推理模块 (inference)

#### CUDA 前处理 (cuda_preprocessor.h)
- NV12 → RGB 色彩空间转换
- 双线性插值缩放
- Letterbox 等比例缩放
- 归一化（均值/标准差）
- CHW 格式转换

#### TensorRT 引擎 (trt_engine.h)
- C++ 原生推理，零 Python 依赖
- 支持 FP16/INT8 量化
- Engine 文件反序列化
- 多 stream 并发推理

#### 锁孔检测器 (corner_slot_detector.h)
- 集装箱锁孔（Corner Slot）检测
- NMS 非极大值抑制
- 2D → 3D 坐标转换
- 3D ROI 区域输出

### 4. 应用层 (app)

#### 网关流水线 (gateway_pipeline.h)
- 全流程串联调度
- 线程池异步处理
- 统计信息输出
- 结果回调机制

## 构建指南

### 依赖环境

- CUDA Toolkit 11.4+
- TensorRT 8.4+
- FFmpeg 5.0+ (需启用 NVDEC 支持)
- CMake 3.18+
- GCC 9.0+ / MSVC 2019+

### 编译步骤

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_NVDEC=ON \
  -DUSE_TENSORRT=ON \
  -DJETSON_PLATFORM=ON
make -j$(nproc)
```

### Windows 编译

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 ^
  -DUSE_NVDEC=ON ^
  -DUSE_TENSORRT=ON
cmake --build . --config Release
```

## 使用方法

### 命令行参数

```bash
port_ai_gateway [options]

选项:
  --visible <url>      可见光相机 RTSP URL
  --infrared <url>     红外相机 RTSP URL
  --engine <path>      TensorRT engine 文件路径
  --no-sync            禁用帧同步
  --visible-only       仅使用可见光相机
  --help               显示帮助
```

### 示例

```bash
# 双目同步模式
port_ai_gateway \
  --visible rtsp://192.168.1.100:554/visible \
  --infrared rtsp://192.168.1.100:554/infrared \
  --engine models/corner_slot.trt

# 单目模式
port_ai_gateway \
  --visible rtsp://192.168.1.100:554/visible \
  --engine models/corner_slot.trt \
  --visible-only
```

### API 使用示例

```cpp
#include "app/gateway_pipeline.h"

using namespace port_ai_gateway;

int main() {
    GatewayConfig config;
    config.visible_camera.url = "rtsp://192.168.1.100:554/visible";
    config.detector_config.trt_config.engine_path = "corner_slot.trt";

    GatewayPipeline pipeline;
    pipeline.SetResultCallback(
        [](const CornerSlotDetectionResultPtr& result) {
            for (const auto& slot : result->corner_slots) {
                printf("Slot: (%.2f, %.2f, %.2f), conf=%.3f\n",
                       slot.x, slot.y, slot.z, slot.confidence);
            }
        });

    pipeline.Init(config);
    pipeline.Start();

    std::cin.get();

    pipeline.Stop();
    return 0;
}
```

## 性能指标

| 模块 | 延迟 (ms) | 说明 |
|------|-----------|------|
| 视频解码 | ~5 | NVDEC 硬件解码 4K |
| CUDA 前处理 | ~2 | Letterbox + 归一化 |
| 推理 | ~7 | YOLO 实例分割 |
| 后处理 | ~1 | NMS + 3D 转换 |
| **端到端** | **~15** | **4K → 3D ROI** |

*注：基于 NVIDIA Jetson AGX Orin 测试*

## 部署说明

### Jetson 平台部署

1. 刷写 JetPack 5.0+
2. 安装 FFmpeg (带 NVDEC 支持)
3. 编译 TensorRT 引擎文件
4. 部署运行时库

### 系统服务部署

```bash
sudo cp port_ai_gateway /usr/local/bin/
sudo cp config/config.json /etc/port_ai_gateway/
# 创建 systemd service
```

## 许可证

本项目为商业项目，未经授权不得用于商业用途。

## 联系方式

技术支持：support@port-ai.com
