import json
import os
from collections import defaultdict


def bio_to_json_with_id(input_path, output_path):
    """
    将BIO格式的NER标注数据转换为带编号的JSON格式

    参数:
        input_path: 输入文件路径（BIO格式）
        output_path: 输出文件路径（JSON格式）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件 {input_path} 不存在")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"开始转换文件: {input_path}")

    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        samples = [s.strip() for s in content.split('\n\n') if s.strip()]

    print(f"共找到 {len(samples)} 个样本")

    # 存储转换后的结果
    json_data = []

    for sample_id, sample in enumerate(samples, 1):  # 从1开始编号
        # 存储当前样本的实体和文本
        entities = []
        tokens = []
        char_pos = 0  # 当前字符位置
        current_entity = None

        for line in sample.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    token, tag = parts[0], parts[1]
                    tokens.append(token)

                    # 处理实体标签
                    if tag.startswith('B_'):
                        # 结束上一个实体（如果有）
                        if current_entity:
                            entities.append(current_entity)

                        # 开始新实体
                        entity_type = tag[2:]  # 去掉B_前缀
                        current_entity = {
                            'start': char_pos,
                            'end': char_pos + len(token) - 1,
                            'type': entity_type,
                            'text': token
                        }
                    elif tag.startswith('I_'):
                        # 扩展当前实体
                        if current_entity and tag[2:] == current_entity['type']:
                            current_entity['end'] = char_pos + len(token) - 1
                            current_entity['text'] += token
                        else:
                            # 不匹配的I标签，视为新实体（根据需求调整）
                            if current_entity:
                                entities.append(current_entity)
                            current_entity = {
                                'start': char_pos,
                                'end': char_pos + len(token) - 1,
                                'type': tag[2:],
                                'text': token
                            }
                    else:
                        # O标签，结束当前实体（如果有）
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None

                    # 更新字符位置（考虑中文等非空格分隔语言）
                    char_pos += len(token)

        # 添加最后一个实体（如果有）
        if current_entity:
            entities.append(current_entity)

        # 构建完整文本
        full_text = ''.join(tokens)

        # 添加到结果列表
        json_data.append({
            'id': sample_id,  # 添加句子编号
            'text': full_text,
            'entities': entities,
            'char_length': len(full_text),  # 添加文本长度信息
            'entity_count': len(entities)  # 添加实体数量信息
        })

    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    total_entities = sum(len(item['entities']) for item in json_data)
    print(f"转换完成！结果已保存到: {output_path}")
    print(f"统计信息:")
    print(f"- 总样本数: {len(json_data)}")
    print(f"- 总实体数: {total_entities}")
    print(f"- 平均每句实体数: {total_entities / len(json_data):.2f}")
    print(f"- 最长文本长度: {max(item['char_length'] for item in json_data)} 字符")
    print(f"- 最短文本长度: {min(item['char_length'] for item in json_data)} 字符")


# 使用您的实际路径
input_path = r""
output_path = r""  # 建议使用.json扩展名

# 执行转换
bio_to_json_with_id(input_path, output_path)