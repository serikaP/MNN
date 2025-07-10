import codecs
import re


def convert_audio_encoding_to_codecs_format(input_file, output_file, target_shape):
    """
    将音频编码结果文件转换为codecs.txt的格式

    Args:
        input_file: 输入文件路径 (audio_encoding_result格式)
        output_file: 输出文件路径 (codecs.txt格式)
        target_shape: 目标输出形状 (batch_size, channels, sequence_length)
    """

    # 读取输入文件中的编码数据
    encodings = []
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # 跳过注释行，提取数字编码
        for line in lines:
            line = line.strip()
            # 跳过注释行和空行
            if line.startswith('#') or not line:
                continue
            # 检查是否为纯数字
            if line.isdigit():
                encodings.append(int(line))

    print(f"读取到 {len(encodings)} 个编码")

    # 根据目标形状重新组织数据
    batch_size, channels, seq_length = target_shape
    total_expected = batch_size * channels * seq_length

    if len(encodings) != total_expected:
        print(f"警告: 编码数量 {len(encodings)} 与期望的 {total_expected} 不匹配")
        # 如果数据不足，用0填充；如果数据过多，截断
        if len(encodings) < total_expected:
            encodings.extend([0] * (total_expected - len(encodings)))
        else:
            encodings = encodings[:total_expected]

    # 将一维数据重新组织为指定形状
    result_arrays = []
    idx = 0

    for batch in range(batch_size):
        batch_data = []
        for channel in range(channels):
            channel_data = []
            for seq in range(seq_length):
                if idx < len(encodings):
                    channel_data.append(encodings[idx])
                    idx += 1
                else:
                    channel_data.append(0)
            batch_data.append(channel_data)
        result_arrays.append(batch_data)

    # 写入输出文件，使用codecs.txt的格式
    with codecs.open(output_file, 'w', encoding='utf-8') as f:
        f.write("example [")

        for i, batch in enumerate(result_arrays):
            # 每个batch作为一个数组
            for j, channel in enumerate(batch):
                f.write("[")
                # 写入通道数据
                for k, value in enumerate(channel):
                    f.write(str(value))
                    if k < len(channel) - 1:
                        f.write(", ")
                f.write("]")

                # 如果不是最后一个元素，添加逗号和换行
                if not (i == len(result_arrays) - 1 and j == len(batch) - 1):
                    f.write(", ")

        f.write("]\n")

    print(f"转换完成，输出形状: {target_shape}")
    print(f"结果已保存到: {output_file}")


def extract_shape_from_comments(input_file):
    """
    从输入文件的注释中提取形状信息
    """
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# 输出形状:'):
                # 提取形状信息，如 [32, 1, 126]
                shape_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', line)
                if shape_match:
                    return (int(shape_match.group(1)),
                            int(shape_match.group(2)),
                            int(shape_match.group(3)))
    return (32, 1, 126)  # 默认形状


# 使用示例
if __name__ == "__main__":
    input_file = "android_encoding_result.txt"
    output_file = "converted_android_codecs.txt"

    # 从输入文件中提取形状信息
    target_shape = extract_shape_from_comments(input_file)
    print(f"检测到目标形状: {target_shape}")

    # 执行转换
    convert_audio_encoding_to_codecs_format(input_file, output_file, target_shape)

    # 验证输出
    print("\n验证输出文件前几行:")
    with codecs.open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content[:200] + "...")