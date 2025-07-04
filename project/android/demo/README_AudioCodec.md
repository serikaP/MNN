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
    输入: waveform [B, 1, T] 
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
            wav = wav.unsqueeze(1)  # (B,1,T)
        
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
# 进入工作目录
cd funcodec
mkdir exp && cd exp
# 下载模型
git clone https://www.modelscope.cn/iic/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch.git
# 模型转换
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
# codecs.txt用于验证模型正确性
```

---
### ONNX模型转换为MNN模型

详见MNN官方文档 ( https://mnn-docs.readthedocs.io/en/latest/tools/convert.html )

从源码编译MNN或从pip安装MNN得到MNNConvert

这里给出从pip install MNN的例子：

```bash
pip install MNN
# 模型转换
mnnconvert -f ONNX --modelFile funcodec_encoder.onnx --MNNModel funcodec_encoder.mnn --bizCode biz
mnnconvert -f ONNX --modelFile funcodec_decoder.onnx --MNNModel funcodec_decoder.mnn --bizCode biz
```
## Android AudioCodec功能



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



#### 内存管理
- **模型缓存**：首次加载后缓存复用
- **音频缓存**：导入文件自动清理
- **结果存储**：应用私有目录，避免权限问题

---

## 项目文件结构

```
MNN-3.2.0/project/android/demo/
├── funcodec/
│   ├── export_funcodec_to_onnx.py        # 编码器转换脚本
│   ├── export_funcodec_decoder_to_onnx.py # 解码器转换脚本
│   └── test_mnn.py                       # PC端测试脚本
└── app/src/main/
    ├── java/com/taobao/android/
    │   ├── mnndemo/
    │   │   └── AudioCodecActivity.java    # 主Activity（编码+解码）
    │   └── utils/
    │       ├── AudioFileReader.java       # 音频文件解析
    │       ├── AudioPlayer.java           # 音频播放和保存
    │       └── WavFileGenerator.java      # WAV文件生成
    ├── res/
    │   ├── layout/activity_audio_codec.xml # 界面布局（编码+解码UI）
    │   └── xml/file_paths.xml             # FileProvider配置
    └── assets/AudioCodec/
        ├── funcodec_encoder.mnn           # 编码器MNN模型
        ├── funcodec_decoder.mnn           # 解码器MNN模型
        └── example.wav                    # 示例音频

```

---

## 使用指南

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
git clone https://github.com/serikaP/MNN.git
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



---

## 性能基准

### 测试环境
- **设备**：Xiaomi MI 8
- **模型**：audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch（转换后148MB）
- **输入**：5.02秒16kHz单声道音频（80,000采样点）

### 性能结果

#### 编码性能
```
=== 性能指标 ===
音频时长: 5.00 秒
编码时间: 8019.15 ms
实时率: 0.63x

=== 编码结果 ===
输出形状: [32, 1, 126]
编码数量: 4032
编码范围: [0, 1023]
压缩比: 19.8:1
```

#### 解码性能
```
=== 性能指标 ===
解码时间: 24616.85 ms
音频时长: 5.04 秒
实时率: 0.20x

=== 解码结果 ===
输入形状: [1, 32, 126]
输出形状: [1, 1, 80640]
输出数据长度: 80640
```

---


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

## 开发日志

### 主要里程碑

1. **模型转换** 
   - 完成FunCodec → ONNX → MNN转换
   - 在PC端验证了模型推理正确性

2. **用户界面完成** 
   - 实现音频导入、编码、导出全流程
   - 但是安卓端的编码与linux上的编码不同

3. **Android集成完成** 
   - MNN引擎成功集成
   - 解决了动态输入shape问题

4. **解码功能完成** 
   - 成功添加FunCodec解码器支持
   - 实现完整的编码-解码流水线
   - 集成音频播放和保存功能

---

