#!/usr/bin/env python3
"""
MNN解码器模型正确性校验脚本
用法: python test_mnn_decoder.py --mnn_path funcodec_decoder.mnn --codes_path result_example.txt
"""

import argparse
import numpy as np
import MNN.expr as F
import time
import os
import re

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
                        help='MNN解码器模型文件路径')
    parser.add_argument('--codes_path', required=True,
                        help='编码文件路径')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='输出音频采样率')
    parser.add_argument('--benchmark', action='store_true',
                        help='是否进行性能测试')
    parser.add_argument('--iterations', type=int, default=50,
                        help='性能测试迭代次数')
    parser.add_argument('--audio_output', default=None,
                        help='输出音频文件路径（默认自动生成）')
    parser.add_argument('--force_shape', default=None,
                        help='强制指定输入形状，格式：B,Q,T 例如：1,32,268')
    return parser.parse_args()


def load_codes_from_file(codes_path, force_shape=None):
    """
    从文件加载编码数据
    
    Args:
        codes_path: 编码文件路径
        force_shape: 强制指定的形状，格式："B,Q,T"
    
    Returns:
        codes_array: 编码数组
        metadata: 元数据信息
    """
    print(f"加载编码文件: {codes_path}")
    
    if not os.path.exists(codes_path):
        raise FileNotFoundError(f"编码文件不存在: {codes_path}")
    
    metadata = {}
    codes_data = []
    
    with open(codes_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # 解析元数据
                if '输出形状:' in line or 'output_shape:' in line:
                    # 提取形状信息
                    shape_match = re.search(r'\[([^\]]+)\]', line)
                    if shape_match:
                        shape_str = shape_match.group(1)
                        try:
                            metadata['shape'] = [int(x.strip()) for x in shape_str.split(',')]
                        except:
                            pass
                elif '编码数量:' in line:
                    # 提取编码数量
                    count_match = re.search(r'编码数量:\s*(\d+)', line)
                    if count_match:
                        metadata['count'] = int(count_match.group(1))
                elif '推理时间:' in line:
                    # 提取推理时间
                    time_match = re.search(r'推理时间:\s*([\d.]+)', line)
                    if time_match:
                        metadata['inference_time'] = float(time_match.group(1))
                continue
            
            if line and not line.startswith('#'):
                try:
                    # 尝试解析为整数
                    codes_data.append(int(float(line)))
                except ValueError:
                    # 如果是多个值在一行，按空格分割
                    values = line.split()
                    for val in values:
                        try:
                            codes_data.append(int(float(val)))
                        except ValueError:
                            continue
    
    codes_array = np.array(codes_data, dtype=np.int32)
    print(f"加载了 {len(codes_data)} 个编码值")
    
    # 确定形状
    if force_shape:
        # 使用强制指定的形状
        try:
            target_shape = [int(x.strip()) for x in force_shape.split(',')]
            expected_size = np.prod(target_shape)
            if len(codes_data) != expected_size:
                print(f"警告: 编码数量({len(codes_data)})与指定形状({target_shape})不匹配(期望{expected_size})")
                # 截断或填充
                if len(codes_data) > expected_size:
                    codes_array = codes_array[:expected_size]
                    print(f"截断到 {expected_size} 个编码")
                else:
                    padding = np.zeros(expected_size - len(codes_data), dtype=np.int32)
                    codes_array = np.concatenate([codes_array, padding])
                    print(f"填充到 {expected_size} 个编码")
            
            codes_array = codes_array.reshape(target_shape)
            metadata['shape'] = target_shape
            print(f"使用强制指定的形状: {target_shape}")
        except Exception as e:
            print(f"强制形状解析失败: {e}")
            force_shape = None
    
    if not force_shape:
        # 自动推断形状
        if 'shape' in metadata:
            target_shape = metadata['shape']
            expected_size = np.prod(target_shape)
            
            if len(codes_data) == expected_size:
                codes_array = codes_array.reshape(target_shape)
                print(f"使用元数据中的形状: {target_shape}")
            else:
                print(f"警告: 编码数量({len(codes_data)})与元数据形状({target_shape})不匹配(期望{expected_size})")
                # 尝试推断为 [1, 32, frames]
                if len(codes_data) % 32 == 0:
                    frames = len(codes_data) // 32
                    target_shape = [1, 32, frames]
                    codes_array = codes_array.reshape(target_shape)
                    metadata['shape'] = target_shape
                    print(f"自动推断形状为: {target_shape}")
                else:
                    print(f"无法推断合适的形状，使用原始一维数组: {codes_array.shape}")
        else:
            # 没有元数据，尝试推断
            if len(codes_data) % 32 == 0:
                frames = len(codes_data) // 32
                target_shape = [1, 32, frames]
                codes_array = codes_array.reshape(target_shape)
                metadata['shape'] = target_shape
                print(f"自动推断形状为: {target_shape}")
            else:
                print(f"无法推断形状，使用原始形状: {codes_array.shape}")
    
    print(f"最终编码数组形状: {codes_array.shape}")
    print(f"编码范围: [{codes_array.min()}, {codes_array.max()}]")
    
    return codes_array, metadata


def save_audio(audio_array, output_path, sample_rate=16000):
    """
    保存音频文件
    
    Args:
        audio_array: 音频数组
        output_path: 输出文件路径
        sample_rate: 采样率
    """
    if sf is None:
        raise RuntimeError("需要安装 soundfile 库: pip install soundfile")
    
    # 确保音频数据是正确的格式
    if audio_array.ndim > 1:
        # 如果是多维，取第一个通道或平均
        if audio_array.shape[0] == 1:
            audio_array = audio_array[0]  # (1, T) -> (T,)
        elif audio_array.shape[1] == 1:
            audio_array = audio_array[:, 0]  # (T, 1) -> (T,)
        else:
            audio_array = audio_array.mean(axis=0)  # 多通道平均
    
    # 确保是一维数组
    audio_array = audio_array.flatten()
    
    # 归一化到 [-1, 1] 范围
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
        print(f"音频已归一化，原始最大值: {max_val:.3f}")
    
    # 保存为wav文件
    sf.write(output_path, audio_array, sample_rate)
    
    duration = len(audio_array) / sample_rate
    print(f"音频已保存到: {output_path}")
    print(f"音频时长: {duration:.2f}秒")
    print(f"采样率: {sample_rate}Hz")
    print(f"音频范围: [{audio_array.min():.3f}, {audio_array.max():.3f}]")


def test_mnn_decoder(mnn_path, codes_path, sample_rate=16000, force_shape=None, audio_output=None):
    """测试MNN解码器模型的正确性"""
    print(f"加载MNN解码器模型: {mnn_path}")

    # 加载模型
    vars_dict = F.load_as_dict(mnn_path)
    print(f"模型包含的变量: {list(vars_dict.keys())}")

    # 获取输入输出变量
    input_name = 'codes'
    output_name = 'waveform'

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

    # 加载编码数据
    codes_array, metadata = load_codes_from_file(codes_path, force_shape)

    print(f"\n输入编码数据形状: {codes_array.shape}")

    # 检查模型输入形状
    model_input_shape = input_var.shape
    print(f"MNN模型期望的输入形状: {model_input_shape}")
    
    # 调整输入变量的形状（如果需要）
    if input_var.shape != list(codes_array.shape):
        print(f"调整MNN输入变量形状: {input_var.shape} -> {list(codes_array.shape)}")
        input_var.resize(codes_array.shape)

    # 写入数据
    print(f"\n最终写入MNN的数据形状: {codes_array.shape}")
    print(f"前10个编码值: {codes_array.flatten()[:10]}")
    print(f"后10个编码值: {codes_array.flatten()[-10:]}")
    print(f"\n写入输入数据...")

    input_var.write(codes_array.tolist())

    # 执行推理
    print(f"执行解码推理...")
    start_time = time.time()
    result = output_var.read()
    inference_time = time.time() - start_time

    # 输出结果
    result_array = np.array(result)
    print(f"\n解码完成!")
    print(f"推理时间: {inference_time * 1000:.2f} ms")
    print(f"输出形状: {result_array.shape}")
    print(f"输出数据类型: {result_array.dtype}")
    print(f"输出范围: [{result_array.min():.3f}, {result_array.max():.3f}]")
    
    # 计算音频时长和实时率
    if result_array.ndim > 1:
        audio_samples = result_array.shape[-1]  # 最后一维是时间维度
    else:
        audio_samples = len(result_array)
    
    audio_duration = audio_samples / sample_rate
    real_time_ratio = audio_duration * 1000 / (inference_time * 1000)
    print(f"输出音频时长: {audio_duration:.2f}s")
    print(f"实时率: {real_time_ratio:.2f}x")
    
    # 保存音频文件
    if audio_output is None:
        # 自动生成输出文件名
        base_name = os.path.splitext(os.path.basename(codes_path))[0]
        timestamp = int(time.time() * 1000)
        audio_output = f"decoded_{base_name}_{timestamp}.wav"
    
    save_audio(result_array, audio_output, sample_rate)
    
    # 保存详细结果文件
    result_filename = f"decoder_result_{os.path.splitext(os.path.basename(codes_path))[0]}.txt"
    with open(result_filename, 'w') as f:
        f.write(f"# MNN解码器推理结果\n")
        f.write(f"# 输入编码文件: {codes_path}\n")
        f.write(f"# 输入形状: {codes_array.shape}\n")
        f.write(f"# 输出形状: {result_array.shape}\n")
        f.write(f"# 输出数据类型: {result_array.dtype}\n")
        f.write(f"# 推理时间: {inference_time * 1000:.2f} ms\n")
        f.write(f"# 音频时长: {audio_duration:.2f}s\n")
        f.write(f"# 实时率: {real_time_ratio:.2f}x\n")
        f.write(f"# 输出音频文件: {audio_output}\n")
        f.write("# 音频数据统计:\n")
        f.write(f"# 最小值: {result_array.min():.6f}\n")
        f.write(f"# 最大值: {result_array.max():.6f}\n")
        f.write(f"# 均值: {result_array.mean():.6f}\n")
        f.write(f"# 标准差: {result_array.std():.6f}\n")

    print(f"详细结果已保存到: {result_filename}")

    return {
        'output': result_array,
        'inference_time': inference_time,
        'input_shape': codes_array.shape,
        'output_shape': result_array.shape,
        'audio_duration': audio_duration,
        'real_time_ratio': real_time_ratio,
        'audio_file': audio_output
    }


def benchmark_decoder(mnn_path, codes_path, iterations=50, sample_rate=16000, force_shape=None):
    """解码器性能基准测试"""
    print(f"\n开始解码器性能测试，迭代次数: {iterations}")

    # 加载模型和数据
    vars_dict = F.load_as_dict(mnn_path)
    input_var = vars_dict['codes']
    output_var = vars_dict['waveform']

    codes_array, _ = load_codes_from_file(codes_path, force_shape)

    # 调整输入形状
    if input_var.shape != list(codes_array.shape):
        input_var.resize(codes_array.shape)

    input_var.write(codes_array.tolist())

    # 预热
    print("预热中...")
    for _ in range(5):
        _ = output_var.read()

    # 正式测试
    print("开始计时测试...")
    times = []

    for i in range(iterations):
        start_time = time.time()
        result = output_var.read()
        end_time = time.time()

        times.append((end_time - start_time) * 1000)  # 转换为毫秒

        if (i + 1) % 10 == 0:
            print(f"完成 {i + 1}/{iterations} 次迭代")

    # 统计结果
    times = np.array(times)
    
    # 计算音频时长
    result_array = np.array(result)
    if result_array.ndim > 1:
        audio_samples = result_array.shape[-1]
    else:
        audio_samples = len(result_array)
    audio_duration = audio_samples / sample_rate

    print(f"\n解码器性能测试结果:")
    print(f"输出音频时长: {audio_duration:.3f} 秒")
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

    print("=" * 60)
    print("FunCodec MNN解码器模型测试")
    print("=" * 60)

    # 基本正确性测试
    result = test_mnn_decoder(args.mnn_path, args.codes_path, args.sample_rate, 
                            args.force_shape, args.audio_output)

    if result is None:
        print("解码器测试失败")
        return

    # 性能测试
    if args.benchmark:
        benchmark_decoder(args.mnn_path, args.codes_path, args.iterations, 
                        args.sample_rate, args.force_shape)

    print("\n解码器测试完成!")
    print(f"输出音频文件: {result['audio_file']}")


if __name__ == '__main__':
    main() 