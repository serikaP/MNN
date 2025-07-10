#!/usr/bin/env python3
"""
MNN模型正确性校验脚本
用法: python test_mnn.py --mnn_path funcodec_encoder.mnn --wav_path example.wav
"""

import argparse
import numpy as np
import MNN.expr as F
import time
import os

# 可选依赖
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnn_path', required=True,
                        help='MNN模型文件路径')
    parser.add_argument('--wav_path', default='example.wav',
                        help='输入音频文件路径')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='目标采样率')
    parser.add_argument('--benchmark', action='store_true',
                        help='是否进行性能测试')
    parser.add_argument('--iterations', type=int, default=100,
                        help='性能测试迭代次数')
    parser.add_argument('--save_codes', action='store_true',
                        help='是否保存编码文件（Android兼容格式）')
    parser.add_argument('--codes_output', default=None,
                        help='编码文件输出路径（默认自动生成）')
    return parser.parse_args()


def load_audio(wav_path, target_sr=16000):
    """加载音频文件并重采样到目标采样率"""
    if sf is None:
        raise RuntimeError("需要安装 soundfile 库: pip install soundfile")

    if not os.path.exists(wav_path):
        print(f"警告: 未找到音频文件 {wav_path}，使用随机噪声")
        # 生成1秒随机噪声
        wav = np.random.randn(target_sr).astype(np.float32)
        return wav

    wav, sr_file = sf.read(wav_path)

    # 如果采样率不匹配，进行重采样
    if sr_file != target_sr:
        if librosa is None:
            raise RuntimeError("需要安装 librosa 库进行重采样: pip install librosa")
        wav = librosa.resample(y=wav, orig_sr=sr_file, target_sr=target_sr)

    # 确保是float32格式
    wav = wav.astype(np.float32)

    # 如果是立体声，转为单声道
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    print(f"加载音频: {wav_path}, 长度: {len(wav) / target_sr:.2f}s, 采样率: {target_sr}Hz")
    return wav


def save_codes_android_format(codes_array, output_path, input_wav_path, inference_time, output_shape):
    """
    保存编码文件为Android应用兼容的格式
    
    Args:
        codes_array: 编码数组
        output_path: 输出文件路径
        input_wav_path: 输入音频文件路径
        inference_time: 推理时间（秒）
        output_shape: 输出形状
    """
    timestamp = str(int(time.time() * 1000))
    
    with open(output_path, 'w') as f:
        # 写入头部信息
        f.write("# FunCodec 编码结果\n")
        f.write(f"# 时间戳: {timestamp}\n")
        f.write(f"# 输入文件: {input_wav_path}\n")
        f.write(f"# 输出形状: {list(output_shape)}\n")
        f.write(f"# 编码数量: {codes_array.size}\n")
        f.write(f"# 推理时间: {inference_time * 1000:.2f} ms\n")
        f.write("# 编码数据:\n")
        
        # 写入编码数据，每行一个值
        flat_codes = codes_array.flatten()
        for code in flat_codes:
            f.write(f"{int(code)}\n")
    
    print(f"编码文件已保存到: {output_path}")
    print(f"编码数量: {codes_array.size}")
    print(f"文件格式: Android兼容格式")


def test_mnn_model(mnn_path, wav_path, sample_rate=16000, save_codes=False, codes_output=None):
    """测试MNN模型的正确性"""
    print(f"加载MNN模型: {mnn_path}")

    # 加载模型
    vars_dict = F.load_as_dict(mnn_path)
    print(f"模型包含的变量: {list(vars_dict.keys())}")

    # 获取输入变量
    input_name = 'waveform'
    output_name = 'codes'

    if input_name not in vars_dict:
        print(f"错误: 模型中未找到输入变量 '{input_name}'")
        print(f"可用变量: {list(vars_dict.keys())}")
        return None

    if output_name not in vars_dict:
        print(f"错误: 模型中未找到输出变量 '{output_name}'")
        print(f"可用变量: {list(vars_dict.keys())}")
        return None

    input_var = vars_dict[input_name]
    output_var = vars_dict[output_name]

    # 查看输入信息
    print(f"\n输入信息:")
    print(f"  形状: {input_var.shape}")
    print(f"  数据格式: {input_var.data_format}")
    print(f"  数据大小: {input_var.size}")

    # 加载音频数据
    wav_data = load_audio(wav_path, sample_rate)

    # 准备输入数据：添加batch维度 (1, T)
    input_data = wav_data.reshape(1, -1).astype(np.float32)
    print(f"\n输入数据形状: {input_data.shape}")

    # 如果模型输入shape固定，可能需要调整输入长度
    model_input_shape = input_var.shape
    print(f"MNN模型期望的输入形状 (来自input_var.shape): {model_input_shape}")
    if len(model_input_shape) >= 2 and model_input_shape[1] > 0:
        expected_length = model_input_shape[1]
        print(f"模型期望的固定长度: {expected_length}")
        if input_data.shape[1] != expected_length:
            print(f"调整输入长度从 {input_data.shape[1]} 到 {expected_length}")
            if input_data.shape[1] > expected_length:
                input_data = input_data[:, :expected_length]
            else:
                # 零填充
                padding = np.zeros((1, expected_length - input_data.shape[1]), dtype=np.float32)
                input_data = np.concatenate([input_data, padding], axis=1)
            print(f"调整后的输入数据形状: {input_data.shape}")
        else:
            print(f"输入数据长度 {input_data.shape[1]} 与模型期望长度 {expected_length} 一致，无需调整。")
    else:
        print(
            f"模型输入长度是动态的 (model_input_shape[1] = {model_input_shape[1] if len(model_input_shape) >= 2 else 'N/A'})。使用原始输入长度 {input_data.shape[1]}。")

    # 调整输入变量的形状（如果需要）
    if input_var.shape != list(input_data.shape):
        print(f"调整MNN输入变量形状: {input_var.shape} -> {list(input_data.shape)}")
        input_var.resize(input_data.shape)

    # 写入数据
    print(f"\n最终写入MNN的数据形状: {input_data.shape}")
    print(f"前5个样本值: {input_data.flatten()[:5]}")
    print(f"后5个样本值: {input_data.flatten()[-5:]}")
    print(f"\n写入输入数据...")

    input_var.write(input_data.tolist())

    # 执行推理
    print(f"执行推理...")
    start_time = time.time()
    result = output_var.read()
    inference_time = time.time() - start_time

    # 输出结果
    result_array = np.array(result)
    print(f"\n推理完成!")
    print(f"推理时间: {inference_time * 1000:.2f} ms")
    print(f"输出形状: {result_array.shape}")
    print(f"输出数据类型: {result_array.dtype}")
    print(f"输出范围: [{result_array.min()}, {result_array.max()}]")
    
    # 计算实时率
    audio_duration = len(wav_data) / sample_rate
    real_time_ratio = audio_duration * 1000 / (inference_time * 1000)
    print(f"音频时长: {audio_duration:.2f}s")
    print(f"实时率: {real_time_ratio:.2f}x")
    
    # 保存编码文件（Android兼容格式）
    if save_codes:
        if codes_output is None:
            # 自动生成输出文件名
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            timestamp = int(time.time() * 1000)
            codes_output = f"codes_{base_name}_{timestamp}.txt"
        
        save_codes_android_format(result_array, codes_output, wav_path, inference_time, result_array.shape)
    
    # 保存详细结果文件
    output_filename = f"result_{os.path.splitext(os.path.basename(wav_path))[0]}.txt"
    with open(output_filename, 'w') as f:
        f.write(f"# MNN模型推理结果\n")
        f.write(f"# 输入文件: {wav_path}\n")
        f.write(f"# 输出形状: {result_array.shape}\n")
        f.write(f"# 数据类型: {result_array.dtype}\n")
        f.write(f"# 推理时间: {inference_time * 1000:.2f} ms\n")
        f.write(f"# 音频时长: {audio_duration:.2f}s\n")
        f.write(f"# 实时率: {real_time_ratio:.2f}x\n")
        f.write("# 结果数据:\n")

        if result_array.ndim == 1:
            # 一维数组，每行一个值
            for value in result_array:
                f.write(f"{value}\n")
        elif result_array.ndim == 2:
            # 二维数组，每行一个向量
            for row in result_array:
                f.write(" ".join(map(str, row)) + "\n")
        else:
            # 多维数组，展平保存
            flat_array = result_array.flatten()
            for value in flat_array:
                f.write(f"{value}\n")

    print(f"详细结果已保存到: {output_filename}")
    
    # 打印部分输出值
    if result_array.size > 0:
        flat_result = result_array.flatten()
        print(f"前20个输出值: {flat_result[:20]}")
        if len(flat_result) > 20:
            print(f"后20个输出值: {flat_result[-20:]}")

    return {
        'output': result_array,
        'inference_time': inference_time,
        'input_shape': input_data.shape,
        'output_shape': result_array.shape,
        'audio_duration': audio_duration,
        'real_time_ratio': real_time_ratio
    }


def benchmark_model(mnn_path, wav_path, iterations=100, sample_rate=16000):
    """性能基准测试"""
    print(f"\n开始性能测试，迭代次数: {iterations}")

    # 加载模型和数据
    vars_dict = F.load_as_dict(mnn_path)
    input_var = vars_dict['waveform']
    output_var = vars_dict['codes']

    wav_data = load_audio(wav_path, sample_rate)
    input_data = wav_data.reshape(1, -1).astype(np.float32)

    # 调整输入形状
    if input_var.shape != list(input_data.shape):
        input_var.resize(input_data.shape)

    input_var.write(input_data.tolist())

    # 预热
    print("预热中...")
    for _ in range(10):
        _ = output_var.read()

    # 正式测试
    print("开始计时测试...")
    times = []

    for i in range(iterations):
        start_time = time.time()
        result = output_var.read()
        end_time = time.time()

        times.append((end_time - start_time) * 1000)  # 转换为毫秒

        if (i + 1) % 20 == 0:
            print(f"完成 {i + 1}/{iterations} 次迭代")

    # 统计结果
    times = np.array(times)
    audio_duration = len(wav_data) / sample_rate  # 音频时长（秒）

    print(f"\n性能测试结果:")
    print(f"音频时长: {audio_duration:.3f} 秒")
    print(f"平均推理时间: {times.mean():.2f} ms")
    print(f"最小推理时间: {times.min():.2f} ms")
    print(f"最大推理时间: {times.max():.2f} ms")
    print(f"标准差: {times.std():.2f} ms")
    print(f"首响延迟 (第一次): {times[0]:.2f} ms")
    print(f"实时率: {audio_duration * 1000 / times.mean():.2f}x")

    # 计算百分位数
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        print(f"P{p}: {np.percentile(times, p):.2f} ms")


def main():
    args = parse_args()

    print("=" * 50)
    print("FunCodec MNN编码器模型测试")
    print("=" * 50)

    # 基本正确性测试
    result = test_mnn_model(args.mnn_path, args.wav_path, args.sample_rate, 
                          args.save_codes, args.codes_output)

    if result is None:
        print("模型测试失败")
        return

    # 性能测试
    if args.benchmark:
        benchmark_model(args.mnn_path, args.wav_path, args.iterations, args.sample_rate)

    print("\n测试完成!")


if __name__ == '__main__':
    main()