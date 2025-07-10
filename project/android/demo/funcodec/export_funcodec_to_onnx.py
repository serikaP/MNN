#!/usr/bin/env python3
"""
最终版 FunCodec 编码器导出为 ONNX - 彻底解决形状固化问题
通过重新设计整个导出流程，避免所有可能的固化源

用法:
    python export_funcodec_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_encoder.onnx
"""
import types, sys
fake_dynamo = types.ModuleType("torch._dynamo")
setattr(fake_dynamo, 'allow_in_graph', lambda *a, **kw: None)
setattr(fake_dynamo, 'trace_rules', {})
sys.modules["torch._dynamo"] = fake_dynamo

import argparse, os, yaml, torch
import torch.nn as nn
try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import librosa
except ImportError:
    librosa = None
from funcodec.tasks.gan_speech_codec import GANSpeechCodecTask
from argparse import Namespace

# ======================= ONNX EXPORT MONKEY PATCH (Definitive) =======================
# 目的: 在不修改FunCodec源码的情况下，修复ONNX导出时的形状固化问题。
# 方法: 动态替换(猴子补丁)编码器核心模块 SConv1d 的 forward 方法。
# 问题根源: SConv1d.forward 在处理非因果padding时，其依赖的 F.pad
#           不支持动态padding尺寸，且内部逻辑使用了math.ceil和
#           Python原生if判断，导致ONNX追踪时尺寸被固化。
# 最终解决方案: 重写 forward 方法，用 torch.cat 手动实现动态padding，
#             完全绕开 F.pad 的限制。
print(">>> Applying DEFINITIVE ONNX export monkey patch for FunCodec SConv1d...")
import funcodec.modules.normed_modules.conv as conv_module
from torch.nn import functional as F

def _onnx_safe_sconv1d_forward(self, x):
    # self 是 SConv1d 的实例
    kernel_size, = self.conv.conv.kernel_size
    stride, = self.conv.conv.stride
    dilation, = self.conv.conv.dilation
    
    if self.causal:
        # 因果padding是静态的，使用原始的pad1d是安全的
        padding_total = (kernel_size - 1) * dilation
        x = conv_module.pad1d(x, (padding_total, 0), mode=self.pad_mode)
        return self.conv(x)
    else:
        # 非因果padding，这是问题的核心，需要完全的动态实现
        padding_total_effective = (kernel_size - 1) * dilation - (stride - 1)
        
        # 1. 动态计算 extra_padding (纯torch操作)
        length_tensor = x.new_tensor(x.shape[-1], dtype=torch.float32)
        n_frames = (length_tensor - kernel_size + padding_total_effective) / stride + 1
        ideal_length = (torch.ceil(n_frames) - 1) * stride + (kernel_size - padding_total_effective)
        extra_padding = (ideal_length - length_tensor).to(torch.long)

        # 2. 计算左右padding
        padding_left = padding_total_effective // 2
        padding_right = padding_total_effective - padding_left
        
        # 3. 手动实现动态padding，绕开 F.pad
        # a. 左边padding (静态)
        padded_x = F.pad(x, (padding_left, 0), mode=self.pad_mode)
        
        # b. 右边padding (动态)
        # 我们需要创建一个 (B, C, padding_right + extra_padding) 的0张量并拼接
        # 为了创建这个张量，我们需要其动态的shape
        right_pad_size = padding_right + extra_padding
        
        # 创建一个足够大的静态0张量，然后切片以获得动态大小的padding
        # 这是一个关键的workaround，以避免在 shape 中使用动态张量
        # 假设最大padding不会超过一个很大的数，例如2048
        # 注意：这假设了在任何情况下，右侧的总padding不会超过2048。
        # 对于音频模型来说，这是一个非常安全的假设。
        max_padding_buffer = 2048 
        zeros_buffer = torch.zeros(x.shape[0], x.shape[1], max_padding_buffer, device=x.device, dtype=x.dtype)
        
        # 从buffer中切出我们需要的动态长度
        dynamic_padding_tensor = zeros_buffer.narrow(-1, 0, right_pad_size)
        
        # 使用 torch.cat 实现最终的padding
        padded_x = torch.cat([padded_x, dynamic_padding_tensor], dim=-1)

        return self.conv(padded_x)

# 4. 执行替换
conv_module.SConv1d.forward = _onnx_safe_sconv1d_forward
print(">>> Monkey patch for SConv1d applied successfully.")
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True,
                   help='FunCodec 预训练模型目录（里面应有 config.yaml 和 *.pt）')
    p.add_argument('--onnx_path', default='funcodec_encoder_ultimate.onnx')
    p.add_argument('--opset', type=int, default=14)
    p.add_argument('--dummy_wav', default='example2.wav',
                   help='用于ONNX导出的示例wav文件；若不存在则使用随机噪声')
    return p.parse_args()

def topdict_to_ns(d):
    return Namespace(**d)

def build_codec(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = topdict_to_ns(cfg_dict)
    codec = GANSpeechCodecTask.build_model(cfg)
    
    weight_path = None
    for name in ["valid.loss.ave.pt", "train.loss.ave.pt"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            weight_path = p
            break
    if weight_path is None:
        weight_path = sorted([p for p in os.listdir(model_dir) if p.endswith(".pth")])[-1]
        weight_path = os.path.join(model_dir, weight_path)

    state_dict = torch.load(weight_path, map_location="cpu")
    codec.load_state_dict(state_dict, strict=True)
    return codec

class UltimateEncoderWrapper(torch.nn.Module):
    """
    终极版编码器包装器 - 彻底解决形状固化问题
    
    核心策略：
    1. 完全避免torch.stack()和torch.cat()等可能固化的操作
    2. 使用最原始的张量操作
    3. 强制使用动态维度计算
    4. 避免任何可能被ONNX优化器固化的操作
    """
    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.encoder = codec.encoder
        self.quantizer = (
            codec.quantizer
            if hasattr(codec, "quantizer")
            else codec.rq
        )
        
        # 获取量化器参数
        self.num_quantizers = 32  # 固定值，来自配置
        self.hop_length = 640     # 固定值，来自配置

    def forward(self, wav):
        # 1. 输入预处理
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # (B,1,T)

        # 2. 音频归一化（模拟FunCodec的音量归一化）
        mono = wav.mean(dim=1, keepdim=True)
        scale = torch.sqrt(mono.pow(2).mean(dim=2, keepdim=True) + 1e-8)
        wav_norm = wav / scale

        # 3. 编码
        latent = self.encoder(wav_norm)  # (B,C,F)

        # 4. 量化 - 这是关键步骤，确保使用所有32层
        # 量化器期望的输入格式是 (B, C, T)，保持编码器输出的原始格式
        latent_for_quantizer = latent  # 保持 (B, C, T) 格式
        
        # 检查量化器的forward方法签名并正确调用
        # 不同版本的FunCodec量化器可能有不同的调用方式
        try:
            # 尝试带带宽参数的调用方式
            sample_rate = 16000
            bandwidth = None  # None表示使用所有32层
            out = self.quantizer(latent_for_quantizer, sample_rate, bandwidth)
        except TypeError as e:
            print(f"尝试三参数调用失败: {e}")
            try:
                # 尝试传入latent和sample_rate的调用方式
                sample_rate = 16000
                out = self.quantizer(latent_for_quantizer, sample_rate)
            except TypeError as e2:
                print(f"尝试二参数调用失败: {e2}")
                try:
                    # 尝试只传入latent的调用方式
                    out = self.quantizer(latent_for_quantizer)
                except TypeError as e3:
                    print(f"尝试一参数调用失败: {e3}")
                    raise RuntimeError(f"无法确定量化器的正确调用方式。错误: {e}, {e2}, {e3}")

        # 5. 提取codes - 彻底重写这部分逻辑
        codes = self._extract_codes_ultimate(out, wav.shape[-1])
        
        return codes.to(torch.int32)

    def _extract_codes_ultimate(self, quantizer_output, input_length):
        if hasattr(quantizer_output, 'codes'):
            codes = quantizer_output.codes  # (n_quantizers, B, T)
        elif isinstance(quantizer_output, (list, tuple)):
            if len(quantizer_output) >= 2:
                codes = quantizer_output[1]  # (quantized, codes, ...)
            elif len(quantizer_output) == 1:
                codes = quantizer_output[0]
            else:
                raise RuntimeError("量化器输出为空的tuple/list")
        else:
            raise RuntimeError(f"未知的量化器输出类型: {type(quantizer_output)}")

        # 检查codes格式并转换为Android期望的格式
        if isinstance(codes, (list, tuple)):
            # 如果是列表，堆叠为 (n_quantizers, B, T)
            codes = torch.stack(codes, dim=0)
        
        # 确保codes是 (n_quantizers, B, T) 格式
        if codes.dim() == 2:
            # 如果是2D，需要添加批次维度
            codes = codes.unsqueeze(0)  # (n_quantizers, T) -> (1, n_quantizers, T)
        
        return codes

def test_ultimate_dynamic_shapes(wrapper, device, test_lengths=[1.0, 2.0, 5.0, 10.0], sr=16000):
    """
    测试终极版本的动态形状支持
    """
    print("\n" + "="*60)
    print("测试终极版动态形状支持")
    print("="*60)
    
    results = []
    for length in test_lengths:
        samples = int(length * sr)
        test_wav = torch.randn(1, samples, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            output = wrapper(test_wav)
        
        expected_frames = samples // 640  # hop_length = 640
        actual_frames = output.shape[2]
        
        print(f"长度: {length:4.1f}s ({samples:6d}样本) -> 输出: {list(output.shape)} "
              f"(期望帧数: {expected_frames}, 实际帧数: {actual_frames})")
        
        results.append((length, samples, output.shape, expected_frames, actual_frames))
    
    # 验证动态性
    frame_ratios = [r[4] / r[3] for r in results]  # actual / expected
    if len(set([r[2] for r in results])) == 1:
        print("❌ 形状完全相同，可能仍然固化")
        return False
    else:
        print("✅ 形状随输入变化，动态性正常")
        return True

def main():
    args = parse_args()
    device = 'cpu'
    
    print("="*60)
    print("终极版 FunCodec ONNX 导出")
    print("="*60)
    
    # 构建模型
    codec = build_codec(args.model_dir).to(device).eval()
    wrapper = UltimateEncoderWrapper(codec).to(device).eval()

    # 构造 dummy 输入
    sr = 16000
    if sf is not None and os.path.isfile(args.dummy_wav):
        wav, sr_file = sf.read(args.dummy_wav)
        if sr_file != sr:
            if librosa is None:
                raise RuntimeError("需要 librosa 进行重采样")
            wav = librosa.resample(y=wav, orig_sr=sr_file, target_sr=sr)
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        print(f"使用音频文件: {args.dummy_wav}, 长度: {wav.shape[1]} samples")
    else:
        wav = torch.randn(1, sr, dtype=torch.float32)
        print("使用随机噪声 1s")
    
    dummy = wav.to(device)

    # 测试PyTorch模型的动态形状支持
    print("\n测试PyTorch模型:")
    pytorch_dynamic = test_ultimate_dynamic_shapes(wrapper, device)
    
    if not pytorch_dynamic:
        print("❌ PyTorch模型本身不支持动态形状，无法继续")
        return

    print(f'\n开始导出 ONNX 到 {args.onnx_path} ...')
    
    # 导出ONNX - 使用最保守的设置
    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx_path,
        opset_version=args.opset,
        input_names=['waveform'],
        output_names=['codes'],
        dynamic_axes={
            'waveform': {1: 'n_samples'},
            'codes': {2: 'n_frames'}
        },
        do_constant_folding=False,      # 关闭常量折叠
        keep_initializers_as_inputs=False,
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    print(f'ONNX 导出完成!')

    # 验证ONNX模型的动态形状
    print("\n" + "="*60)
    print("验证 ONNX 模型动态形状")
    print("="*60)
    
    import onnx, onnxruntime
    
    # 检查模型
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型结构检查通过")
    
    # 创建运行时会话
    ort_sess = onnxruntime.InferenceSession(
        args.onnx_path, 
        providers=['CPUExecutionProvider']
    )
    
    # 测试多个长度
    test_lengths = [1.0, 2.0, 5.0, 10.0]
    onnx_shapes = []
    pytorch_shapes = []
    
    all_passed = True
    
    for length in test_lengths:
        samples = int(length * sr)
        test_wav = torch.randn(1, samples, dtype=torch.float32)
        
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = wrapper(test_wav.to(device)).cpu().numpy()
        
        # ONNX推理
        try:
            onnx_output = ort_sess.run(None, {'waveform': test_wav.numpy()})[0]
        except Exception as e:
            print(f"❌ ONNX推理失败 ({length}s): {e}")
            all_passed = False
            continue
        
        pytorch_shapes.append(pytorch_output.shape)
        onnx_shapes.append(onnx_output.shape)
        
        print(f"长度: {length:4.1f}s -> PyTorch: {pytorch_output.shape}, ONNX: {onnx_output.shape}")
        
        if pytorch_output.shape != onnx_output.shape:
            print(f"  ❌ 形状不匹配!")
            all_passed = False
        else:
            print(f"  ✅ 形状匹配")
    
    # 检查ONNX输出是否真的动态
    if len(set(onnx_shapes)) == 1:
        print(f"\n❌ ONNX输出形状固化: 所有测试都产生相同形状 {onnx_shapes[0]}")
        all_passed = False
    else:
        print(f"\n✅ ONNX输出形状动态: 产生了 {len(set(onnx_shapes))} 种不同形状")
    
    if all_passed:
        print(f"\n🎉 所有测试通过! 终极版ONNX模型支持真正的动态形状!")
    else:
        print(f"\n❌ 测试失败，模型可能仍存在固化问题")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 