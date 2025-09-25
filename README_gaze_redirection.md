# Gaze Redirection 测试

## 项目概述

基于VAE-Gaze模型的注视重定向系统，能够将连续视频帧中的注视方向重定向到摄像头方向，并输出对比GIF动图。

## Input

- 一段连续frames（视频文件）
- 支持格式：MP4, AVI, MOV等常见视频格式

## Output

- 重定向眼神角度到摄像头视角的GIF动图演示
- 左右对比：原始视频 vs 注视重定向后的视频

## Pipeline

```
input_frame → face landmarks detection (face_alignment) → eye mask face image → model prediction → output
```

### 详细流程

1. **视频输入**：读取连续的视频帧
2. **人脸检测**：使用face_alignment检测面部关键点（68点）
3. **眼部遮罩**：基于关键点生成精确的眼部遮罩
4. **模型推理**：VAE+UNet+GazeNet处理注视重定向
5. **GIF生成**：创建左右对比的动图输出

## 快速开始

### 环境准备

```bash
# 激活conda环境
conda activate your_env_name

# 安装依赖包
pip install torch torchvision opencv-python face_alignment pillow imageio omegaconf diffusers
```

### 基本使用

```bash
# 基础命令
python run_video_gaze.py --input video.mp4 --checkpoint model.pth

# 自定义输出
python run_video_gaze.py --input video.mp4 --checkpoint model.pth \
                         --output result.gif --max-frames 30 --fps 10
```

### 演示Demo

```bash
# 运行完整演示
python demo_video_gaze.py --run-demo

# 查看使用示例
python demo_video_gaze.py --show-examples
```

## 文件结构

```
├── video_gaze_redirection.py    # 核心视频处理模块
├── run_video_gaze.py           # 简化运行脚本
├── demo_video_gaze.py          # 演示和测试脚本
├── prediction.py               # 单图片处理模块（兼容性）
├── configs/
│   ├── training/stage2.yaml        # 训练配置
│   └── inference/gaze_redirection.yaml  # 推理配置
├── models/                     # 模型文件
├── checkpoints/               # 训练检查点
└── output/                   # 输出结果
```

## 技术特性

- **高质量重定向**：基于VAE+UNet架构的注视方向精确重定向
- **实时人脸检测**：使用face_alignment进行鲁棒的面部关键点检测
- **智能眼部遮罩**：自动生成精确的眼部遮罩，支持fallback机制
- **GIF动图输出**：生成便于分享和展示的动图格式
- **批量处理**：支持多帧连续处理和批量视频处理
- **设备兼容**：支持CUDA GPU加速和CPU后备处理

## 参数配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入视频文件路径 | 必须 |
| `--checkpoint` | 模型检查点路径 | 自动检测 |
| `--output` | 输出GIF路径 | `./output/gaze_redirection.gif` |
| `--max-frames` | 最大处理帧数 | 50 |
| `--fps` | 输出GIF帧率 | 10 |
| `--device` | 计算设备 | `cuda` |

## 使用示例

### 1. 基本使用

```bash
python run_video_gaze.py --input selfie_video.mp4 --checkpoint model.pth
```

### 2. 高质量输出

```bash
python run_video_gaze.py --input video.mp4 --checkpoint model.pth \
                         --max-frames 60 --fps 15 --output hq_result.gif
```

### 3. CPU处理

```bash
python run_video_gaze.py --input video.mp4 --checkpoint model.pth --device cpu
```

## 输出说明

执行完成后，输出GIF文件包含：

- **左侧**：原始视频帧，显示原本的注视方向
- **右侧**：处理后的帧，注视方向已重定向至摄像头方向
- **对比效果**：清晰展示注视重定向的效果

## 故障排除

### 常见问题

1. **face_alignment模型下载问题**

   ```bash
   # face_alignment会在首次使用时自动下载模型
   # 如果下载失败，可以手动安装：
   pip install --upgrade face_alignment

   # 或者设置镜像源（中国用户）：
   pip install face_alignment -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

2. **CUDA内存不足**

   ```bash
   # 使用CPU处理
   python run_video_gaze.py --input video.mp4 --checkpoint model.pth --device cpu
   ```

3. **找不到检查点文件**
   - 确保已完成模型训练
   - 检查检查点文件路径
   - 使用 `--checkpoint` 参数显式指定路径

### 性能优化

- **GPU加速**：使用CUDA设备可显著提升处理速度
- **帧数限制**：通过 `--max-frames` 控制处理的帧数
- **分辨率调整**：系统自动将帧调整为适合的分辨率

## 技术原理

### 模型架构

- **VAE编码器**：将图像编码到潜空间
- **GazeNet MLP**：编码头部姿态和目标注视方向
- **UNet网络**：融合视觉特征和注视嵌入
- **VAE解码器**：生成注视重定向后的图像

### 注视坐标系

- **目标方向**：[0.0, 0.0] 表示直视摄像头
- **坐标系统**：使用pitch和yaw表示注视方向
- **单位**：弧度制

## 扩展功能

### Python API调用

```python
from video_gaze_redirection import VideoGazeRedirectionProcessor

# 初始化处理器
processor = VideoGazeRedirectionProcessor(
    config_path="configs/training/stage2.yaml",
    checkpoint_path="checkpoints/latest.pth"
)

# 处理视频
success = processor.process_video_to_gif(
    input_path="input.mp4",
    output_path="output.gif",
    max_frames=40,
    fps=12
)
```

### 批量处理

```python
import os

for video_file in os.listdir("input_videos/"):
    if video_file.endswith('.mp4'):
        processor.process_video_to_gif(
            input_path=f"input_videos/{video_file}",
            output_path=f"output/{video_file.replace('.mp4', '.gif')}"
        )
```

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。
