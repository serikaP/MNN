#!/usr/bin/env python3
"""
将 FunCodec 解码器导出为 ONNX
用法:
    python export_funcodec_decoder_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_decoder.onnx --opset 14 --simplify --dummy_codes_path codecs.txt

"""
import types, sys
fake_dynamo = types.ModuleType("torch._dynamo")
fake_dynamo.allow_in_graph = lambda *args, **kwargs: lambda x: x
sys.modules["torch._dynamo"] = fake_dynamo
import argparse, os, yaml, torch
import numpy as np
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True,
                   help='FunCodec 预训练模型目录（里面应有 config.yaml 和 *.pt）')
    p.add_argument('--onnx_path', default='funcodec_decoder.onnx')
    p.add_argument('--opset', type=int, default=14)
    p.add_argument('--simplify', action='store_true',
                   help='导出后调用 onnxsim 简化图')
    p.add_argument('--dummy_codes_path', default='',
                   help='用于ONNX导出的示例编码文件路径')
    return p.parse_args()

def topdict_to_ns(d):
    return Namespace(**d)

def build_codec(model_dir):
    # 读取配置
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = topdict_to_ns(cfg_dict)
    codec = GANSpeechCodecTask.build_model(cfg)
    
    # 载入权重（优先使用 *.ave.pt）
    weight_path = None
    for name in ["valid.loss.ave.pt", "train.loss.ave.pt"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            weight_path = p; break
    if weight_path is None:
        weight_path = sorted([p for p in os.listdir(model_dir) if p.endswith(".pth")])[-1]
        weight_path = os.path.join(model_dir, weight_path)

    state_dict = torch.load(weight_path, map_location="cpu")
    codec.load_state_dict(state_dict, strict=True)
    return codec

class DecoderWrapper(torch.nn.Module):
    """
    基于FunCodec源码的标准解码流程
    输入: codes [B, n_q, frames] -> 转换为 [n_q, B, frames] -> 解码 -> 输出: waveform [B, 1, T]
    """
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, codes):
        """
        输入: codes [B, n_q, frames] (int32/long)
        输出: waveform [B, 1, T] (float32)
        """
        codes = codes.to(torch.long)
        
        # 根据FunCodec源码，解码器期望的输入格式是 [n_q, B, frames]
        # 但我们的输入是 [B, n_q, frames]，需要转换
        codes = codes.permute(1, 0, 2)  # [B, n_q, frames] -> [n_q, B, frames]
        
        # 使用量化器解码
        # 根据源码，quantizer.decode期望输入是 [n_q, B, frames] 格式
        quantized = self.codec.quantizer.decode(codes)  # 输出: [B, D, frames]
        
        # 使用解码器重建音频
        # quantized需要转换为 [B, frames, D] 格式给解码器
        quantized = quantized.transpose(1, 2)  # [B, D, frames] -> [B, frames, D]
        
        # 创建EncodedFrame格式的输入
        # 根据源码，_decode_frame期望的是 (code_embs, scale) 元组
        codes_with_scale = [(quantized, None)]  # scale设为None，表示不使用音量缩放
        
        # 调用解码器
        waveform = self.codec._decode(codes_with_scale)
        
        # 确保输出格式正确 [B, 1, T]
        if waveform.dim() == 2:  # [B, T]
            waveform = waveform.unsqueeze(1)  # [B, 1, T]
        elif waveform.dim() == 3 and waveform.shape[1] != 1:  # [B, C, T] where C != 1
            waveform = waveform.mean(dim=1, keepdim=True)  # 转为单声道
        
        return waveform

def generate_dummy_codes(shape=(1, 32, 126)):
    """生成示例编码数据用于ONNX导出"""
    # 生成随机的量化索引，范围通常是0-1023（10位量化）
    codes = np.random.randint(0, 1024, size=shape, dtype=np.int32)
    return torch.from_numpy(codes)

def load_codes_from_file(file_path):
    """从文件加载编码数据"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 跳过注释行，提取编码数据
        codes_data = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    codes_data.append(int(line))
                except ValueError:
                    continue  # 跳过无法解析的行
        
        # 根据FunCodec标准格式重塑数据
        if len(codes_data) > 0:
            # 确保数据长度是32的倍数
            if len(codes_data) % 32 != 0:
                print(f"警告: 编码数据长度 {len(codes_data)} 不是32的倍数，截断数据")
                codes_data = codes_data[:len(codes_data) // 32 * 32]
            
            total_frames = len(codes_data) // 32
            print(f"加载编码数据: {len(codes_data)} 个值, 重塑为 [1, 32, {total_frames}]")
            
            # 重塑为 [batch=1, n_q=32, frames]
            codes_array = np.array(codes_data, dtype=np.int32).reshape(1, 32, total_frames)
            return torch.from_numpy(codes_array)
        else:
            print("编码文件中没有找到有效数据")
            return None
    except Exception as e:
        print(f"加载编码文件失败: {e}")
        return None

def main():
    args = parse_args()
    device = 'cpu'
    codec = build_codec(args.model_dir).to(device).eval()
    wrapper = DecoderWrapper(codec).to(device).eval()

    # 构造 dummy 输入：优先使用指定编码文件
    if args.dummy_codes_path and os.path.isfile(args.dummy_codes_path):
        dummy_codes = load_codes_from_file(args.dummy_codes_path)
        if dummy_codes is not None:
            print(f"使用编码文件 {args.dummy_codes_path} 作为 dummy，形状: {dummy_codes.shape}")
        else:
            dummy_codes = generate_dummy_codes()
            print("编码文件加载失败，使用随机编码数据")
    else:
        dummy_codes = generate_dummy_codes()
        print("使用随机编码数据作为 dummy，形状: (1, 32, 126)")
    
    dummy = dummy_codes.to(device)

    print('开始导出解码器 ONNX …')
    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx_path,
        opset_version=args.opset,
        input_names=['codes'],
        output_names=['waveform'],
        dynamic_axes={'codes': {2: 'n_frames'},
                      'waveform': {2: 'n_samples'}},
        do_constant_folding=False,
    )
    print(f'解码器 ONNX 导出完成 => {args.onnx_path}')

    # 基本正确性校验
    import onnx, onnxruntime
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = onnxruntime.InferenceSession(args.onnx_path,
                                            providers=['CPUExecutionProvider'])
    ort_out = ort_sess.run(None, {'codes': dummy.cpu().numpy()})[0]
    print('解码器 ONNX 运行成功，输出形状：', ort_out.shape)

    # 对比 PyTorch 与 ONNX 输出
    with torch.no_grad():
        pt_out = wrapper(dummy).cpu().numpy()

    print("PyTorch 输出形状:", pt_out.shape)
    print("ONNX 输出形状:", ort_out.shape)

    if pt_out.shape == ort_out.shape:
        diff = np.abs(pt_out - ort_out).max()
        print(f"PyTorch 与 ONNX 输出最大差异: {diff}")
        if diff < 1e-5:
            print("PyTorch 与 ONNX 输出基本一致")
        else:
            print("PyTorch 与 ONNX 输出存在差异")

    # 保存测试输出音频
    if sf is not None:
        output_audio = ort_out.squeeze()  # 移除batch和channel维度
        sf.write('decoded_test_output.wav', output_audio, 16000)
        print('测试解码音频已保存: decoded_test_output.wav')

    # （可选）onnxsim 简化
    if args.simplify:
        print('调用 onnxsim 简化 …')
        from onnxsim import simplify
        model_simp, ok = simplify(onnx_model,
                                  overwrite_input_shapes={'codes': list(dummy.shape)})
        assert ok, 'Simplify Failed!'
        onnx.save(model_simp, args.onnx_path.replace('.onnx', '_sim.onnx'))
        print('简化后解码器模型已保存。')

if __name__ == '__main__':
    main() 