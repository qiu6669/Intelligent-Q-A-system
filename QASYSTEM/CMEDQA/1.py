
import random
import os
from collections import defaultdict


def process_dataset(input_path, output_path, sample_size=100, min_length=6):


    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        sample_size: 需要抽取的样本数量
        min_length: 最小文本长度要求(中文字符)
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件 {input_path} 不存在")
        return

    # 创建输出目录(如果不存在)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"开始处理文件: {input_path}")
    print(f"参数设置: 样本数={sample_size}, 最小长度={min_length}")

    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        samples = [s.strip() for s in content.split('\n\n') if s.strip()]

    print(f"原始数据集共 {len(samples)} 个样本")

    # 预处理：仅筛选长度（去除实体数量限制）
    valid_samples = []

    for sample in samples:
        # 提取原始文本（不再统计实体数量）
        tokens = []
        for line in sample.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:  # 需要token和标签
                    token = parts[0]
                    tokens.append(token)

        raw_text = ''.join(tokens)

        # 筛选条件：仅保留长度足够的样本
        if len(raw_text) >= min_length:
            valid_samples.append(sample)

    print(f"符合要求的样本共 {len(valid_samples)} 个 (长度≥{min_length})")

    if len(valid_samples) < sample_size:
        print(f"警告: 有效样本数({len(valid_samples)})小于要求的样本数({sample_size})")
        sample_size = len(valid_samples)

    # 直接随机抽样
    sampled_data = random.sample(valid_samples, sample_size)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(sampled_data))

    print(f"\n处理完成！共筛选出 {len(sampled_data)} 条样本")
    print(f"结果已保存到: {output_path}")


# 使用示例
input_path = r""
output_path = r""

# 处理数据集（去除了 max_entities 限制）
process_dataset(
    input_path=input_path,
    output_path=output_path,
    sample_size=100,
    min_length=6  # 
)