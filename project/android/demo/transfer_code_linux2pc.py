import re

def convert_file(input_filename="codecs.txt", output_filename="codecs_output.txt"):
    """
    读取输入文件，将其中的数字提取出来，并写入到输出文件中，每个数字占一行。

    :param input_filename: 包含原始数据的文件名 (图1格式)
    :param output_filename: 用于保存结果的文件名 (图2格式)
    """
    try:
        # 1. 读取输入文件的全部内容
        with open(input_filename, "r", encoding="utf-8") as f_in:
            content = f_in.read()

        # 2. 使用正则表达式查找所有数字
        # \d+ 匹配一个或多个数字
        numbers = re.findall(r'\d+', content)

        # 3. 将数字列表用换行符连接成一个字符串
        output_content = "\n".join(numbers)

        # 4. 将结果写入输出文件
        with open(output_filename, "w", encoding="utf-8") as f_out:
            f_out.write(output_content)

        print(f"转换成功！结果已保存到 '{output_filename}' 文件中。")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_filename}'。请确保文件存在且路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 调用转换函数，使用默认的文件名 "input.txt" 和 "output.txt"
    convert_file()