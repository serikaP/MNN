# FunCodec Android Demo 工作文档

##  项目概述

本项目实现了FunCodec语音编码模型在Android平台的部署，包括模型转换、MNN推理引擎集成和完整的Android应用Demo。

### 核心功能
- **FunCodec模型转换**：将预训练模型的编码器和解码器分别转换为ONNX格式，再转换为MNN格式
- **Android推理**：基于MNN引擎的实时音频编码和解码
- **完整音频流水线**：音频导入 → 编码 → 导出编码 → 导入编码 → 解码 → 播放/保存音频
- **用户界面**：完整的Android应用，支持全流程音频处理功能

### 技术栈
- **模型**: FunCodec (audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch)
- **推理引擎**: MNN
- **平台**: Android (API 21+)
- **语言**: Python (模型转换) + Java (Android应用)

---

##  FunCodec模型转换脚本

### export_funcodec_to_onnx.py

#### 脚本功能
将FunCodec预训练模型转换为ONNX格式，为后续MNN转换做准备。

#### 核心特性
1. **模型包装**：提取编码器部分（waveform → codes）
2. **音量归一化**：实现FunCodec的标准音频预处理
3. **动态输入**：支持任意长度音频输入
4. **格式验证**：确保转换正确性

#### 主要组件

```python
class EncoderWrapper(torch.nn.Module):
    """
    FunCodec编码器包装类
    输入: waveform [B, 1, T] 或 [B, T]
    输出: codes [B, n_q, frames] (int32)
    """
    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.encoder = codec.encoder      # SEANetEncoder
        self.quantizer = codec.quantizer  # ResidualQuantizer

    def forward(self, wav):
        # 音频预处理和编码
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # 确保 (B,1,T) 格式
        
        # RMS音量归一化
        mono = wav.mean(dim=1, keepdim=True)
        scale = torch.sqrt(mono.pow(2).mean(dim=2, keepdim=True) + 1e-8)
        wav_norm = wav / scale
        
        # 编码过程
        latent = self.encoder(wav_norm)
        codes = self.quantizer(latent)
        
        # 处理输出格式
        if isinstance(codes, (list, tuple)):
            codes = torch.stack(codes, dim=1)
        
        return codes.to(torch.int32)
```

#### 使用方法
以 audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch 模型为例
```bash
#进入工作目录
cd funcodec
mkdir exp && cd exp
#下载模型
git clone https://www.modelscope.cn/iic/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch.git
#模型转换
cd ..
python export_funcodec_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_encoder.onnx --opset 14 --simplify

```

#### 转换流程
1. **模型加载**：从config.yaml和权重文件构建完整模型
2. **编码器提取**：包装为独立的编码器模块
3. **ONNX导出**：支持动态输入形状
4. **正确性验证**：对比PyTorch和ONNX输出
5. **可选简化**：使用onnxsim优化计算图

### export_funcodec_decoder_to_onnx.py

#### 脚本功能
将FunCodec解码器转换为ONNX格式，实现编码到音频的逆向转换。

#### 核心特性
1. **解码器包装**：提取解码器部分（codes → waveform）
2. **反量化处理**：将量化编码恢复为连续特征
3. **动态输入**：支持任意长度编码输入
4. **音频重建**：高质量的音频信号重构

#### 主要组件

```python
class DecoderWrapper(torch.nn.Module):
    """
    FunCodec解码器包装类
    输入: codes [B, n_q, frames] (int32)
    输出: waveform [B, 1, T] (float32)
    """
    def __init__(self, codec):
        super().__init__()
        self.decoder = codec.decoder      # SEANetDecoder
        self.quantizer = codec.quantizer  # ResidualQuantizer

    def forward(self, codes):
        # 反量化：codes -> latent
        latent = self.quantizer.decode(codes)
        
        # 解码：latent -> waveform
        waveform = self.decoder(latent)
        
        return waveform
```

#### 使用方法
```bash
python export_funcodec_decoder_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_decoder.onnx --opset 14 --simplify --dummy_codes_path codecs.txt
#codecs.txt用于验证模型正确性
```

---
### ONNX模型转换为MNN模型

详见MNN官方文档 ( https://mnn-docs.readthedocs.io/en/latest/tools/convert.html )

从源码编译MNN或从pip安装MNN得到MNNConvert

这里给出从pip install MNN的例子：

```bash
pip install MNN
#模型转换
mnnconvert -f ONNX --modelFile funcodec_encoder.onnx --MNNModel funcodec_encoder.mnn --bizCode biz
mnnconvert -f ONNX --modelFile funcodec_decoder.onnx --MNNModel funcodec_decoder.mnn --bizCode biz
```
## 📱 Android AudioCodec功能



### 核心功能模块

#### 1. 模型加载系统

**特点**：
- **双模型管理**：同时加载编码器和解码器模型
- **多级回退策略**：Assets → 外部存储 → 缓存复用
- **大文件处理**：支持148MB+模型文件
- **完整性验证**：文件大小和可读性检查
- **容错机制**：单个模型失败不影响另一个模型的使用

```java
private String prepareModel() {
    // 1. 检查缓存中的有效模型文件
    if (cacheModel.exists() && cacheModel.length() > 100000000) {
        return useExistingModel();
    }
    
    // 2. 尝试从Assets复制
    if (!copyFromAssets()) {
        // 3. 回退到外部存储
        return copyFromExternalStorage();
    }
    
    // 4. 创建MNN实例并验证
    return createAndValidateMNNInstance();
}
```

#### 2. 音频文件处理系统

**AudioFileReader核心能力**：
- **多格式支持**：16/24/32位PCM WAV文件
- **自动转换**：多声道→单声道，任意采样率→16kHz
- **数据归一化**：转换为[-1.0, 1.0]浮点范围

```java
// WAV文件解析流程
byte[] wavData → parseWavHeader() → extractPCMData() 
→ channelMixing() → resample() → float[] audioSamples
```

**重采样算法**：
```java
// 线性插值重采样
for (int i = 0; i < outputLength; i++) {
    double srcIndex = i * ratio;
    int index1 = (int) srcIndex;
    int index2 = Math.min(index1 + 1, input.length - 1);
    double fraction = srcIndex - index1;
    
    output[i] = input[index1] * (1 - fraction) + input[index2] * fraction;
}
```

#### 3. MNN推理引擎集成

**推理流程**：
```java
// 1. 动态调整输入张量形状
mInputTensor.reshape(new int[]{1, audioLength});
mSession.reshape(); // 重新计算计算图

// 2. 设置输入数据
mInputTensor.setInputFloatData(audioData);

// 3. 执行推理
long startTime = System.nanoTime();
mSession.run();
long endTime = System.nanoTime();

// 4. 获取输出
int[] codes = mOutputTensor.getIntData();
```

**性能监控**：
- **编码时间**
- **实时率**：音频时长/编码时间
- **压缩比**：原始采样数/编码数量

#### 4. 音频解码系统

**解码流程**：
```java
// 1. 导入编码数据并重塑
int[] codes = loadCodesFromFile();
int[] inputShape = {1, 32, totalFrames};
mDecoderInputTensor.reshape(inputShape);
mDecoderSession.reshape();

// 2. 设置编码数据
mDecoderInputTensor.setInputIntData(codes);

// 3. 执行解码推理
mDecoderSession.run();

// 4. 获取音频波形
float[] waveform = mDecoderOutputTensor.getFloatData();
```

**解码特性**：
- **编码文件解析**：自动解析文本格式的编码文件
- **格式验证**：确保编码数据符合[32, 1, frames]格式
- **高质量重建**：基于FunCodec解码器的音频重构
- **实时播放**：内置AudioTrack播放引擎

#### 5. 用户交互界面

**五个主要功能区域**：

1. **编码区域**
   - 音频导入和编码
   - 实时性能显示
   - 编码结果导出

2. **解码区域**
   - 编码文件导入
   - 一键解码功能
   - 解码性能监控

3. **音频控制**
   - 实时音频播放
   - WAV文件保存
   - 播放状态控制

4. **文件管理**
   - 多格式文件导入
   - 安全文件分享
   - 自动缓存清理

5. **状态监控**
   - 模型加载进度
   - 处理过程状态
   - 错误信息提示

#### 5. 文件导入导出系统

**导入功能**：
```java
// 支持任意音频格式，自动转换为16kHz WAV
Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
intent.setType("audio/*");
startActivityForResult(intent, REQUEST_IMPORT_WAV);
```

**导出功能**：
```java
// 使用FileProvider安全分享
Uri fileUri = FileProvider.getUriForFile(this, 
    getPackageName() + ".fileprovider", file);
    
Intent shareIntent = new Intent(Intent.ACTION_SEND);
shareIntent.putExtra(Intent.EXTRA_STREAM, fileUri);
```

### 性能特性

#### 编码性能指标
- **输入格式**：16kHz单声道浮点音频
- **输出格式**：(n_q, B, frames)整数编码矩阵
- **典型性能**：5秒音频 → 8.3秒编码时间（0.6x实时率）
- **压缩比**：约20:1（80,000采样点 → 4,032编码值）

#### 内存管理
- **模型缓存**：首次加载后缓存复用
- **音频缓存**：导入文件自动清理
- **结果存储**：应用私有目录，避免权限问题

---

## 📁 项目文件结构

```
MNN-3.2.0/project/android/demo/
├── funcodec/
│   ├── export_funcodec_to_onnx.py        # 编码器转换脚本
│   ├── export_funcodec_decoder_to_onnx.py # 解码器转换脚本
│   └── test_mnn.py                       # PC端测试脚本
├── app/src/main/
│   ├── java/com/taobao/android/
│   │   ├── mnndemo/
│   │   │   └── AudioCodecActivity.java    # 主Activity（编码+解码）
│   │   └── utils/
│   │       ├── AudioFileReader.java       # 音频文件解析
│   │       ├── AudioPlayer.java           # 音频播放和保存
│   │       └── WavFileGenerator.java      # WAV文件生成
│   ├── res/
│   │   ├── layout/activity_audio_codec.xml # 界面布局（编码+解码UI）
│   │   └── xml/file_paths.xml             # FileProvider配置
│   └── assets/AudioCodec/
│       ├── funcodec_encoder.mnn           # 编码器MNN模型
│       ├── funcodec_decoder.mnn           # 解码器MNN模型
│       └── example.wav                    # 示例音频
└── resource/model/AudioCodec/
    ├── funcodec_encoder.mnn              # 备用编码器位置
    └── funcodec_decoder.mnn              # 备用解码器位置
```

---

## 🚀 使用指南

### 环境准备
1. **Python环境**（模型转换）
   ```bash
   pip install torch torchaudio onnx onnxsim
   pip install funcodec 
   ```

2. **Android开发环境**
   - Android Studio
   - NDK
   - API Level 21+

### 完整部署流程

#### 步骤1：模型转换
```bash
# 1. 下载MNN
git clone https://github.com/alibaba/MNN.git
cd MNN/project/android/demo/funcodec

# 2. 下载预训练模型
mkdir exp && cd exp
git clone https://www.modelscope.cn/iic/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch.git
cd ..

# 3. 转换编码器为ONNX
python export_funcodec_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_encoder.onnx --opset 14 --simplify

# 解码器转换
python export_funcodec_decoder_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_decoder.onnx --opset 14 --simplify --dummy_codes_path codecs.txt

# 4. 转换为MNN格式
pip install MNN

# 编码器转换
mnnconvert -f ONNX --modelFile funcodec_encoder.onnx --MNNModel funcodec_encoder.mnn --bizCode biz

# 解码器转换
mnnconvert -f ONNX --modelFile funcodec_decoder.onnx --MNNModel funcodec_decoder.mnn --bizCode biz

# （可选）编码器模型正确性校验
python test_mnn.py --mnn_path funcodec_encoder.mnn --wav_path example.wav

# 5. 复制到Android项目
cp funcodec_encoder.mnn ../app/src/main/assets/AudioCodec/
cp funcodec_decoder.mnn ../app/src/main/assets/AudioCodec/
```

#### 步骤2：编译Android应用
```bash
cd MNN-3.2.0/project/android/demo
./gradlew assembleDebug
```

#### 步骤3：安装测试
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 功能使用

#### 编码功能
1. **基础编码测试**
   - 启动应用，等待模型加载完成
   - 点击"开始编码"使用内置示例音频
   - 查看编码结果和性能指标

2. **自定义音频编码**
   - 点击"导入音频"选择WAV文件
   - 系统自动转换为16kHz格式
   - 点击"开始编码"处理自定义音频

3. **编码结果导出**
   - 编码完成后，"导出编码"按钮激活
   - 点击可通过微信、邮件等方式分享编码文件

#### 解码功能
4. **编码文件导入**
   - 点击"导入编码"选择之前导出的编码文件
   - 系统自动验证编码格式和数据完整性
   - 导入成功后"开始解码"按钮激活

5. **音频解码重建**
   - 点击"开始解码"执行解码推理
   - 查看解码性能指标和音频重建质量
   - 解码完成后音频控制按钮激活

6. **解码音频播放和保存**
   - 点击"播放音频"实时播放重建的音频
   - 点击"保存音频"将解码结果保存为WAV文件
   - 支持暂停/停止播放控制

#### 完整流水线测试
7. **端到端测试**
   - 导入音频 → 编码 → 导出编码文件
   - 导入编码文件 → 解码 → 播放/保存音频
   - 对比原始音频和重建音频的质量

---

## 📊 性能基准

### 测试环境
- **设备**：Xiaomi MI 8
- **模型**：audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch（转换后148MB）
- **输入**：5秒16kHz单声道音频（80,000采样点）

### 性能结果

#### 编码性能
```
=== 性能指标 ===
音频时长: 5.00 秒
编码时间: 8322.30 ms
实时率: 0.60x

=== 编码结果 ===
输出形状: [32, 1, 126]
编码数量: 4032
编码范围: [0, 1023]
压缩比: 19.8:1
```

#### 解码性能
```
=== 性能指标 ===
解码时间: ~6500.00 ms
音频时长: 5.00 秒
实时率: ~0.77x

=== 解码结果 ===
输入形状: [1, 32, 126]
输出形状: [1, 1, 80000]
音频样本数: 80000
音频范围: [-0.8, 0.8]
重建质量: 高保真音频重建
```

#### 端到端性能
- **完整流水线时长**：编码(8.3s) + 解码(6.5s) = 14.8s
- **音频质量**：接近原始音频，适合语音和音乐场景
- **压缩效率**：20:1压缩比，显著减少存储空间
---

## 🛠️ 技术细节

### 关键实现要点

1. **动态输入处理**
   ```java
   // MNN要求先reshape tensor，再reshape session
   mInputTensor.reshape(new int[]{1, audioLength});
   mSession.reshape(); 
   ```

2. **权限管理**
   ```java
   // 使用MNN标准权限工具
   PermissionUtils.askPermission(this, permissions, requestCode, callback);
   ```

3. **大文件处理**
   ```java
   // 使用应用私有目录避免权限问题
   File outputDir = new File(getExternalFilesDir(null), "results");
   ```

### 常见问题排查

1. **模型加载失败**
   - 检查文件完整性（148MB+）
   - 验证MNN格式正确性
   - 查看logcat详细错误信息

2. **推理崩溃**
   - 确保调用session.reshape()
   - 检查输入数据格式（float32）
   - 验证tensor形状匹配

3. **音频导入失败**
   - 确认WAV格式支持（16/24/32位PCM）
   - 检查文件读取权限
   - 查看AudioFileReader日志


---

## 📝 开发日志

### 主要里程碑

1. **模型转换完成** ✅
   - FunCodec → ONNX → MNN转换链路打通
   - 验证了模型推理正确性

2. **Android集成完成** ✅
   - MNN引擎成功集成
   - 解决了动态输入shape问题

3. **用户界面完成** ✅
   - 实现音频导入、编码、导出全流程
   - 优化用户体验和错误处理

4. **性能优化** ✅
   - 实现准确的性能监控
   - 优化内存使用和文件管理

5. **解码功能完成** ✅
   - 成功添加FunCodec解码器支持
   - 实现完整的编码-解码流水线
   - 集成音频播放和保存功能

### 技术挑战解决

1. **MNN动态输入**：发现需要先reshape tensor再reshape session
2. **权限管理**：采用MNN demo标准的PermissionUtils方案
3. **大文件处理**：使用应用私有目录避免Android权限限制
4. **音频格式兼容**：实现了完整的WAV解析和重采样系统
5. **解码器集成**：解决FunCodec解码器的反量化和音频重建问题
6. **音频播放**：实现基于AudioTrack的实时音频播放系统
7. **双模型管理**：优化编码器和解码器的并行加载和资源管理

---

*本文档记录了FunCodec Android Demo的完整技术实现，包括模型转换、推理集成和应用开发的全流程。*
