#!/usr/bin/env python3
"""
将 FunCodec 编码器导出为 ONNX
用法:
    python export_funcodec_to_onnx.py --model_dir exp/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch --onnx_path funcodec_encoder.onnx --opset 14 --simplify
"""
import types, sys
fake_dynamo = types.ModuleType("torch._dynamo")
fake_dynamo.allow_in_graph = lambda *a, **kw: None
sys.modules["torch._dynamo"] = fake_dynamo
import argparse, os, yaml, torch
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
    p.add_argument('--onnx_path', default='funcodec_encoder.onnx')
    p.add_argument('--opset', type=int, default=14)
    p.add_argument('--simplify', action='store_true',
                   help='导出后调用 onnxsim 简化图')
    p.add_argument('--dummy_wav', default='example.wav',
                   help='用于ONNX导出的示例wav文件；若不存在则使用随机噪声')
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

class EncoderWrapper(torch.nn.Module):
    """
    调用 codec 的组件完成 "wav → codes"：
       waveform --(unsqueeze)--> (B,1,T)
                --encoder--> latent
                --quantizer--> codes
    """
    def __init__(self, codec):
        super().__init__()
        self.codec = codec
        self.encoder = codec.encoder          # SEANetEncoder
        self.quantizer = (
            codec.quantizer
            if hasattr(codec, "quantizer")
            else codec.rq                       # residual quantizer
        )

    def forward(self, wav):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)  # (B,1,T)

        # FunCodec 在 _encode_frame 内部会根据 RMS 做音量归一化 (audio_normalize=True)
        mono = wav.mean(dim=1, keepdim=True)
        scale = torch.sqrt(mono.pow(2).mean(dim=2, keepdim=True) + 1e-8)
        wav_norm = wav / scale

        latent = self.encoder(wav_norm)  # (B,C,F)

        out = self.quantizer(latent)

        # 提取 codes
        if isinstance(out, torch.Tensor):  # 只有 codes
            codes = out
        elif isinstance(out, (list, tuple)):
            if len(out) == 1:
                codes = out[0]
            else:
                codes = out[1]  # (quantized, codes, …)
        else:
            raise RuntimeError(f"未知返回类型: {type(out)}")

        # list/tuple of per-layer codes → stack
        if isinstance(codes, (list, tuple)):
            codes = torch.stack(codes, dim=1)  # (B, n_q, frames)

        return codes.to(torch.int32)

def main():
    args = parse_args()
    device = 'cpu'
    codec = build_codec(args.model_dir).to(device).eval()
    wrapper = EncoderWrapper(codec).to(device).eval()

    # 构造 dummy 输入：优先使用指定 wav
    sr = 16000
    if sf is not None and os.path.isfile(args.dummy_wav):
        wav, sr_file = sf.read(args.dummy_wav)
        if sr_file != sr:
            if librosa is None:
                raise RuntimeError("需要 librosa 进行重采样，请安装 librosa 或提供 16kHz 音频")
            wav = librosa.resample(y=wav, orig_sr=sr_file, target_sr=sr)
        wav = torch.from_numpy(wav).float().unsqueeze(0)  # (1,T)
        print(f"使用 {args.dummy_wav} 作为 dummy，长度: {wav.shape[1]}")
    else:
        wav = torch.randn(1, sr, dtype=torch.float32)
        print("未找到dummy_wav 或 soundfile 库不可用，使用随机噪声 1s")
    dummy = wav.to(device)

    print('开始导出 ONNX …')
    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx_path,
        opset_version=args.opset,
        input_names=['waveform'],
        output_names=['codes'],
        dynamic_axes={'waveform': {1: 'n_samples'},
                      'codes': {2: 'n_frames'}},
        do_constant_folding=False,
    )
    print(f'ONNX 导出完成 => {args.onnx_path}')

    # 基本正确性校验
    import onnx, onnxruntime
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = onnxruntime.InferenceSession(args.onnx_path,
                                            providers=['CPUExecutionProvider'])
    ort_out = ort_sess.run(None, {'waveform': dummy.cpu().numpy()})[0]
    print('ONNX 运行成功，输出形状：', ort_out.shape)

    # 对比 PyTorch 与 ONNX 输出
    with torch.no_grad():
        pt_out = wrapper(dummy).cpu().numpy()

    print("PyTorch 前20个值:", pt_out.flatten()[:20])
    print("ONNX   前20个值:",  ort_out.flatten()[:20])

    if pt_out.shape == ort_out.shape:
        diff = (pt_out != ort_out)
        num_diff = diff.sum()
        print(f"总元素: {pt_out.size}, 不同元素: {num_diff}")
        if num_diff == 0:
            print("PyTorch 与 ONNX 输出完全一致")
        else:
            print("PyTorch 与 ONNX 输出存在差异，最大abs diff: ", float(abs(pt_out - ort_out).max()))

    # （可选）onnxsim 简化
    if args.simplify:
        print('调用 onnxsim 简化 …')
        from onnxsim import simplify
        model_simp, ok = simplify(onnx_model,
                                  overwrite_input_shapes={'waveform': list(dummy.shape)})
        assert ok, 'Simplify Failed!'
        onnx.save(model_simp, args.onnx_path.replace('.onnx', '_sim.onnx'))
        print('简化后模型已保存。')

if __name__ == '__main__':
    main()
    #