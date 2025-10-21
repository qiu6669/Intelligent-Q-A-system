import os
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np


# 固定随机种子
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_all_seeds(42)


# 设置 DeepSeek API 配置
os.environ['DEEPSEEK_API_KEY'] = "" 
os.environ['DEEPSEEK_MODEL'] = ""
os.environ['DEEPSEEK_BASE_URL'] = ""  

# 加载数据集
with open(r'') as f:
    dataset = json.load(f)


client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
)

# 定义实体类型及解释
entity_types = [
    'disease',  # 疾病
    'symptom',  # 症状
    'drug',  # 药物名称
    'body',  # 身体部位
    'treatment',  # 治疗方法 例如消炎药
    'test',  # 检查方法 例如拍片，透视等
    'crowd',  # 人群 例如儿童
    'time',  # 时间 例如周五
    'physiology',  # 生理状况 例如月经 消化等
    'feature',  # 程度 例如严重 轻度等
    'department'  # 科室 例如妇科 性病科等
]


def process_item_with_retry(item, max_retries=3):
    random.seed(42)
    for attempt in range(max_retries):
        try:
            result = process_item(item)
            if result is not None:
                return result
        except Exception as e:
            wait_time = (2 ** attempt) + random.random()
            time.sleep(wait_time)
            print(f"样本 {item.get('id')} 第 {attempt + 1} 次重试...")
    return None


def process_item(item):
    """处理单个数据项"""
    random.seed(42)
    text = item['text']
    true_entities = item['entities']

    prompt = f"""请执行医疗文本的严格实体识别，特别注意以下改进要求：
    #
    #     【核心指令】
    #     1.drug必须是有完整名称的实体例如三九感冒灵，如果是泛指药品例如麻药则标注为treatment
    #     2.尽可能拆分所有可能的实体，不要漏标，例如胃酸识别为胃body，头皮痒拆分为痒symptom
    #     3.要标注出重复文本，例如句子：我大姨得了卵巢癌，现在还不知道该怎么治，一般卵巢癌的早期症状有什么啊，卵巢癌要标注两次 
    #     4.字符位置必须从0开始计数,"(",","等标点符号也占一个字符位置，例如：上海哪里皮肤科好（荨麻疹），这里荨是9，上是0
    #     必须识别所有医疗相关短语，包括：{", ".join(entity_types)}
    #     采用"实体文本 [起-止] @ 类型"格式，严格保持原始文本字符位置
    #     5.上海 嘉定这种地名不需标注，它不是feature
    #     【待分析文本】
    #     {text}
    #
    #     请严格按照上述规则输出所有实体，确保：
    #     1. 无遗漏重复实体
    #     2. 专业术语完整标注
    #     3. 包含所有药物和人群
    #     4. 疙瘩 (symptom)，胆结石 (disease)
    #     5. 标注所有生理现象"""

    try:
        time.sleep(0.3 + random.random() * 0.7)

   
        response = client.chat.completions.create(
            model=os.getenv(''),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0,
        )

        # 解析响应
        predicted_entities = []
        response_text = response.choices[0].message.content

        for line in response_text.split('\n'):
            line = line.strip()
            if line and '[' in line and '] @' in line:
                try:
                    parts = line.split(' [', 1)
                    text_part = parts[0]
                    indices_part = parts[1].split('] @', 1)
                    start, end = map(int, indices_part[0].split('-'))
                    entity_type = indices_part[1].strip()

                    if entity_type in entity_types:
                        predicted_entities.append({
                            'start': start,
                            'end': end,
                            'type': entity_type,
                            'text': text_part
                        })
                except Exception as e:
                    continue

        return {
            'text': text,
            'true_entities': true_entities,
            'pred_entities': predicted_entities,
            'id': item.get('id', 0)
        }
    except Exception as e:
        print(f"处理样本 {item.get('id')} 时出错: {str(e)}")
        raise


def evaluate_ner(dataset, sample_size=100):
    """并行评估函数（完全忽略位置和标签大小写差异）"""
    start_time = time.time()

    stats = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'sample_details': []
    }

    # 标签规范化映射（解决拼写不一致问题）
    tag_normalization = {
        'crowed': 'crowd',
        # 可以添加其他需要规范化的标签映射
    }

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_item_with_retry, item)
                   for item in dataset[:sample_size]]

        results = []
        for future in tqdm(futures, total=sample_size, desc="评估进度"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"处理失败: {str(e)}")

    for result in results:
        text = result['text']
        true_entities = result['true_entities']
        pred_entities = result['pred_entities']

        # 构建标准化实体集合（完全忽略类型）
        true_texts = {ent['text'].strip().lower() for ent in true_entities}
        pred_texts = {ent['text'].strip().lower() for ent in pred_entities}

        # 统计指标
        tp = len(true_texts & pred_texts)  # 交集即为真正例
        fp = len(pred_texts - true_texts)  # 预测有但真实没有
        fn = len(true_texts - pred_texts)  # 真实有但预测没有

        stats['true_positives'] += tp
        stats['false_positives'] += fp
        stats['false_negatives'] += fn

        # 记录详细对比信息（忽略类型）
        missed_entities = [ent for ent in true_entities
                           if ent['text'].strip().lower() not in pred_texts]
        wrong_entities = [ent for ent in pred_entities
                          if ent['text'].strip().lower() not in true_texts]

        # 计算样本级指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        stats['sample_details'].append({
            'id': result['id'],
            'text': text,
            'true_entities': true_entities,
            'pred_entities': pred_entities,
            'missed_entities': missed_entities,
            'wrong_entities': wrong_entities,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # 计算总体指标
    precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (
                                                                                                          stats[
                                                                                                              'true_positives'] +
                                                                                                          stats[
                                                                                                              'false_positives']) > 0 else 0
    recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives']) if (
                                                                                                       stats[
                                                                                                           'true_positives'] +
                                                                                                       stats[
                                                                                                           'false_negatives']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': stats['true_positives'],
        'false_positives': stats['false_positives'],
        'false_negatives': stats['false_negatives'],
        'sample_size': len(results),
        'success_rate': len(results) / sample_size,
        'sample_details': stats['sample_details'],
        'total_time': time.time() - start_time
    }


# 运行评估
print("开始评估---")
results = evaluate_ner(dataset, sample_size=100)

# 打印结果
print("\n评估结果:")
print(f"总耗时: {results['total_time']:.2f}秒")
print(f"处理样本量: {results['sample_size']} (成功率: {results['success_rate']:.1%})")
print(f"精确率 (Precision): {results['precision']:.4f}")
print(f"召回率 (Recall): {results['recall']:.4f}")
print(f"F1分数: {results['f1']:.4f}")
print(f"真正例 (TP): {results['true_positives']}")
print(f"假正例 (FP): {results['false_positives']}")
print(f"假反例 (FN): {results['false_negatives']}")

# 输出前10个样本的详细信息
print("\n样本详细信息:")
for i, sample in enumerate(results['sample_details'][:10]):
    print(f"\n=== 样本 {i + 1} ===")
    print(f"样本ID: {sample['id']}")
    print(f"精确率: {sample['precision']:.2f}")
    print(f"召回率: {sample['recall']:.2f}")
    print(f"F1分数: {sample['f1']:.2f}")
    print(f"文本内容: {sample['text']}")

    print("\n真实实体:")
    for ent in sample['true_entities']:
        print(f"- {ent['text']} ({ent['type']})")

    print("\n预测实体:")
    for ent in sample['pred_entities']:
        print(f"- {ent['text']} ({ent['type']})")

    if sample['missed_entities']:
        print("\n遗漏的实体:")
        for ent in sample['missed_entities']:
            print(f"- {ent['text']} ({ent['type']})")

    if sample['wrong_entities']:
        print("\n错误的实体:")
        for ent in sample['wrong_entities']:
            print(f"- {ent['text']} ({ent['type']})")