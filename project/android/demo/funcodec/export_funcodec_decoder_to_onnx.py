#!/usr/bin/env python3
"""
FunCodec 解码器导出为 ONNX，并增加了从文件解码的功能。
- 模式一 (默认): 导出ONNX模型并验证。
    python export_funcodec_decoder_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch
- 模式二: 从文件读取编码并用PyTorch模型解码为音频。
    python export_funcodec_decoder_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --decode_file result_example.txt --output_wav decoded_audio.wav
- 模式三 (新增): 使用ONNX模型从文件读取编码并解码为音频。
    python export_funcodec_decoder_to_onnx.py --onnx_decode_file result_example.txt --onnx_path funcodec_decoder.onnx --output_wav decoded_onnx_audio.wav
"""

# 解决 einops 与 torch.onnx 导出冲突的猴子补丁
import types, sys

fake_dynamo = types.ModuleType("torch._dynamo")
setattr(fake_dynamo, 'allow_in_graph', lambda _a, *kw: None)
setattr(fake_dynamo, 'trace_rules', {})
sys.modules["torch._dynamo"] = fake_dynamo

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace

try:
    import onnx
    import onnxruntime
except ImportError:
    # 在解码模式下，这些不是必需的
    onnx, onnxruntime = None, None

try:
    from scipy.io import wavfile
except ImportError:
    print("请安装 scipy: pip install scipy")
    sys.exit(1)

# ======================= ONNX EXPORT MONKEY PATCH =======================
print(">>> Applying DEFINITIVE ONNX export monkey patch for FunCodec SConv1d...")
import funcodec.modules.normed_modules.conv as conv_module
from torch.nn import functional as F


def _onnx_safe_sconv1d_forward(self, x):
    kernel_size, = self.conv.conv.kernel_size
    stride, = self.conv.conv.stride
    dilation, = self.conv.conv.dilation

    if self.causal:
        padding_total = (kernel_size - 1) * dilation
        x = conv_module.pad1d(x, (padding_total, 0), mode=self.pad_mode)
        return self.conv(x)
    else:
        padding_total_effective = (kernel_size - 1) * dilation - (stride - 1)
        length_tensor = x.new_tensor(x.shape[-1], dtype=torch.float32)
        n_frames = (length_tensor - kernel_size + padding_total_effective) / stride + 1
        ideal_length = (torch.ceil(n_frames) - 1) * stride + (kernel_size - padding_total_effective)
        extra_padding = (ideal_length - length_tensor).to(torch.long)

        padding_left = padding_total_effective // 2
        padding_right = padding_total_effective - padding_left

        padded_x = F.pad(x, (padding_left, 0), mode=self.pad_mode)
        right_pad_size = padding_right + extra_padding

        max_padding_buffer = 2048
        zeros_buffer = torch.zeros(x.shape[0], x.shape[1], max_padding_buffer, device=x.device, dtype=x.dtype)
        dynamic_padding_tensor = zeros_buffer.narrow(-1, 0, right_pad_size)
        padded_x = torch.cat([padded_x, dynamic_padding_tensor], dim=-1)

        return self.conv(padded_x)


conv_module.SConv1d.forward = _onnx_safe_sconv1d_forward
print(">>> Monkey patch for SConv1d applied successfully.")
# ======================================================================

from funcodec.tasks.gan_speech_codec import GANSpeechCodecTask


def parse_args():
    p = argparse.ArgumentParser(description="FunCodec Decoder ONNX Exporter and File Decoder")
    p.add_argument('--model_dir',
                   help='FunCodec 预训练模型目录 (应包含 config.yaml 和 *.pt/.pth)')

    # ONNX相关参数
    p.add_argument('--onnx_path', default='funcodec_decoder.onnx',
                   help='输出的 ONNX 模型路径')
    p.add_argument('--opset', type=int, default=14,
                   help='ONNX opset 版本')
    p.add_argument('--codebook_size', type=int, default=1024,
                   help='码本大小,用于生成dummy input')

    # 文件解码相关参数
    p.add_argument('--decode_file', type=str, default=None,
                   help='从指定文件读取编码并用PyTorch模型解码为音频')
    p.add_argument('--onnx_decode_file', type=str, default=None,
                   help='从指定文件读取编码并用ONNX模型解码为音频')
    p.add_argument('--output_wav', type=str, default='decoded_from_file.wav',
                   help='解码后的音频输出路径')

    return p.parse_args()


def topdict_to_ns(d):
    return Namespace(**d)


def build_codec(model_dir):
    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = topdict_to_ns(cfg_dict)
    codec = GANSpeechCodecTask.build_model(cfg)

    weight_path = None
    for name in ["model.pth", "valid.loss.ave.pt", "train.loss.ave.pt"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            weight_path = p
            break

    if weight_path is None:
        pt_files = sorted([p for p in os.listdir(model_dir) if p.endswith((".pt", ".pth"))])
        if not pt_files:
            raise RuntimeError(f"在 {model_dir} 中找不到任何 .pt 或 .pth 权重文件")
        weight_path = os.path.join(model_dir, pt_files[-1])

    print(f"使用权重文件: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu")
    codec.load_state_dict(state_dict, strict=True)
    return codec


class DecoderWrapper(nn.Module):
    """
    一个包装器,包含FunCodec的反量化器和解码器。
    """

    def __init__(self, codec):
        super().__init__()
        self.quantizer = codec.quantizer
        self.decoder = codec.decoder
        self.hop_length = np.prod(self.decoder.ratios)

    def forward(self, codes):
        codes_transposed = codes.permute(1, 0, 2)
        dequantized_b_t_c = self.quantizer.decode(codes_transposed)
        dequantized_b_c_t = dequantized_b_t_c.permute(0, 2, 1)
        wav = self.decoder(dequantized_b_c_t)
        return wav


def read_codes_from_file(file_path: str, num_quantizers: int = 32) -> torch.Tensor:
    """
    从文本文件中读取编码。
    文件格式假定为: 所有编码数字以空格或换行符分隔。
    """
    print(f"正在从文件 {file_path} 读取编码...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"编码文件不存在: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    # 将所有空白符（空格、换行、制表符）作为分隔符
    str_codes = content.split()
    try:
        int_codes = [int(c) for c in str_codes]
    except ValueError as e:
        print(f"错误: 文件 {file_path} 包含非整数值。")
        raise e

    codes_np = np.array(int_codes, dtype=np.int64)

    # 验证编码总数是否可以被量化器数量整除
    if codes_np.size % num_quantizers != 0:
        raise ValueError(
            f"编码总数 ({codes_np.size}) 无法被量化器层数 ({num_quantizers}) 整除。"
        )

    # 重塑为 (B, Nq, T) 格式, B=1
    num_frames = codes_np.size // num_quantizers
    codes_reshaped = codes_np.reshape((1, num_quantizers, num_frames))
    print(f"读取了 {codes_np.size} 个编码, 重塑为形状: {codes_reshaped.shape}")

    return torch.from_numpy(codes_reshaped)


def decode_from_file(args, device):
    """
    执行从文件解码并保存为WAV的流程。
    """
    print("\n" + "=" * 60)
    print("模式: 从文件解码 (PyTorch模型)")
    print("=" * 60)

    # 1. 构建并加载模型
    codec = build_codec(args.model_dir).to(device).eval()
    wrapper = DecoderWrapper(codec).to(device).eval()

    # 2. 从文件读取编码
    # en-libritts-16k-nq32ds640 模型，量化器数量为32
    num_quantizers = 32
    try:
        codes_tensor = read_codes_from_file(args.decode_file, num_quantizers)
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 3. 使用PyTorch模型进行解码
    print("正在使用PyTorch模型解码...")
    with torch.no_grad():
        output_wav_tensor = wrapper(codes_tensor.to(device))

    # 4. 保存为WAV文件
    output_wav_np = output_wav_tensor.cpu().numpy().squeeze()
    sample_rate = 16000  # FunCodec libritts 模型的标准采样率

    try:
        wavfile.write(args.output_wav, sample_rate, output_wav_np)
        print(f"\n🎉 PyTorch模型解码成功! 音频已保存到: {args.output_wav}")
    except Exception as e:
        print(f"\n❌ 保存WAV文件失败: {e}")
        sys.exit(1)


def decode_from_file_onnx(args, device):
    """
    使用ONNX模型执行从文件解码并保存为WAV的流程。
    """
    print("\n" + "=" * 60)
    print("模式: 从文件解码 (ONNX模型)")
    print("=" * 60)

    # 检查ONNX相关依赖
    if onnx is None or onnxruntime is None:
        print("错误: ONNX解码模式需要安装 onnx 和 onnxruntime。")
        print("请运行: pip install onnx onnxruntime")
        sys.exit(1)

    # 1. 检查ONNX模型文件是否存在
    if not os.path.exists(args.onnx_path):
        print(f"错误: ONNX模型文件不存在: {args.onnx_path}")
        print("请先导出ONNX模型或指定正确的ONNX模型路径。")
        sys.exit(1)

    # 2. 加载ONNX模型
    print(f"正在加载ONNX模型: {args.onnx_path}")
    try:
        ort_sess = onnxruntime.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
        print("✅ ONNX模型加载成功")
    except Exception as e:
        print(f"❌ ONNX模型加载失败: {e}")
        sys.exit(1)

    # 3. 从文件读取编码
    num_quantizers = 32
    try:
        codes_tensor = read_codes_from_file(args.onnx_decode_file, num_quantizers)
    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 4. 使用ONNX模型进行解码
    print("正在使用ONNX模型解码...")
    try:
        codes_numpy = codes_tensor.numpy()
        onnx_output = ort_sess.run(None, {'codes': codes_numpy})[0]
        print(f"ONNX模型输出形状: {onnx_output.shape}")
    except Exception as e:
        print(f"❌ ONNX模型推理失败: {e}")
        sys.exit(1)

    # 5. 保存为WAV文件
    output_wav_np = onnx_output.squeeze()
    sample_rate = 16000  # FunCodec libritts 模型的标准采样率

    try:
        wavfile.write(args.output_wav, sample_rate, output_wav_np)
        print(f"\n🎉 ONNX模型解码成功! 音频已保存到: {args.output_wav}")
    except Exception as e:
        print(f"\n❌ 保存WAV文件失败: {e}")
        sys.exit(1)

    # 6. 显示一些统计信息
    print(f"解码统计信息:")
    print(f"  - 输入编码帧数: {codes_tensor.shape[2]}")
    print(f"  - 输出音频采样点数: {len(output_wav_np)}")
    print(f"  - 音频时长: {len(output_wav_np) / sample_rate:.2f} 秒")


def main():
    args = parse_args()
    device = 'cpu'

    # 根据命令行参数选择执行模式
    if args.onnx_decode_file:
        # 模式三: 使用ONNX模型从文件解码
        decode_from_file_onnx(args, device)
        return True
    elif args.decode_file:
        # 模式二: 使用PyTorch模型从文件解码
        if args.model_dir is None:
            print("错误: 使用PyTorch模型解码时需要指定 --model_dir 参数")
            sys.exit(1)
        decode_from_file(args, device)
        return True

    # 模式一: 导出ONNX模型并验证
    if args.model_dir is None:
        print("错误: 导出ONNX模型时需要指定 --model_dir 参数")
        sys.exit(1)

    if onnx is None or onnxruntime is None:
        print("错误: ONNX导出模式需要安装 onnx 和 onnxruntime。")
        print("请运行: pip install onnx onnxruntime")
        return False

    print("\n" + "=" * 60)
    print("模式: 导出ONNX模型")
    print("=" * 60)

    # 1. 构建并加载模型
    codec = build_codec(args.model_dir).to(device).eval()
    wrapper = DecoderWrapper(codec).to(device).eval()

    # 2. 准备 dummy input
    n_quantizers = 32
    dummy_frames = 50
    dummy_input = torch.randint(
        0, args.codebook_size,
        (1, n_quantizers, dummy_frames),
        dtype=torch.long, device=device
    )
    print(f"使用 dummy input, 形状: {list(dummy_input.shape)}")

    # 3. 导出 ONNX 模型
    print(f'\n开始导出 ONNX 到 {args.onnx_path} ...')
    torch.onnx.export(
        wrapper, dummy_input, args.onnx_path,
        opset_version=args.opset,
        input_names=['codes'], output_names=['waveform'],
        dynamic_axes={'codes': {2: 'n_frames'}, 'waveform': {2: 'n_samples'}},
        do_constant_folding=True, verbose=False
    )
    print(f'ONNX 导出完成!')

    # 4. 验证 ONNX 模型
    print("\n" + "=" * 60)
    print("验证 ONNX 模型动态形状")
    print("=" * 60)

    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX模型结构检查通过")

    ort_sess = onnxruntime.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])

    test_frames = [50, 125]
    all_passed = True
    onnx_shapes = []
    sample_rate = 16000

    for n_frames in test_frames:
        test_codes = torch.randint(0, args.codebook_size, (1, n_quantizers, n_frames), dtype=torch.long)

        with torch.no_grad():
            pytorch_output = wrapper(test_codes.to(device)).cpu().numpy()

        try:
            onnx_output = ort_sess.run(None, {'codes': test_codes.numpy()})[0]
        except Exception as e:
            print(f"❌ ONNX推理失败 (帧数: {n_frames}): {e}")
            all_passed = False
            continue

        print(f"帧数: {n_frames:4d} -> PyTorch: {pytorch_output.shape}, ONNX: {onnx_output.shape}")

        pytorch_wav_path = f"pytorch_output_{n_frames}frames.wav"
        wavfile.write(pytorch_wav_path, sample_rate, pytorch_output.squeeze())
        print(f"  🎵 已保存PyTorch输出到: {pytorch_wav_path}")

        onnx_wav_path = f"onnx_output_{n_frames}frames.wav"
        wavfile.write(onnx_wav_path, sample_rate, onnx_output.squeeze())
        print(f"  🎵 已保存ONNX输出到: {onnx_wav_path}")

        onnx_shapes.append(onnx_output.shape)

        if pytorch_output.shape != onnx_output.shape:
            print(f"  ❌ 形状不匹配!")
            all_passed = False
        else:
            print("  ✅ 形状匹配")

    if len(set(onnx_shapes)) == 1 and len(test_frames) > 1:
        print(f"\n❌ ONNX输出形状固化: {onnx_shapes[0]}")
        all_passed = False
    else:
        print(f"\n✅ ONNX输出形状动态")

    if all_passed:
        print("\n🎉 所有测试通过!")
        print(f"\n💡 提示: 现在可以使用以下命令测试ONNX模型的文件解码功能:")
        print(
            f"python {sys.argv[0]} --onnx_decode_file your_codes.txt --onnx_path {args.onnx_path} --output_wav onnx_decoded.wav")
    else:
        print("\n❌ 测试失败")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)