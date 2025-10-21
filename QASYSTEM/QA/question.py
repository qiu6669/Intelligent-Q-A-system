from config import *
import Levenshtein
import random
import json
from models.model_ir.predict import predict as ir_predict
from models.model_ner.predict import predict as ner_predict
from py2neo import Graph
import traceback
from openai import OpenAI
import os

class GraphQA():
    def __init__(self, use_llm=True):
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.use_llm = use_llm
        if self.use_llm:
            self.llm_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL")
            )

    def get_fresh_intent(self, text):
        intents = []
        with open(FRESH_INTENT_PATH, encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            t, l = line.strip().split(',')
            score = Levenshtein.ratio(text, t)
            if score > FRESH_THRESHOLD:
                intents.append((score, t, l))
        return sorted(intents, reverse=True)[0][-1] if intents else None

    def parse_fresh_answer(self, text, uuid):
        fresh_intent = self.get_fresh_intent(text)
        if not fresh_intent:
            return None
        if fresh_intent == 'deny':
            self.remove_cache(uuid)
            return random.choice(APOLOGY_ANSWER)
        elif fresh_intent == 'accept':
            cache_data = self.load_cache(uuid)
            template = self.fill_template(cache_data['intent'], cache_data['slots'])
            if not template:
                return random.choice(APOLOGY_ANSWER)
            answer = self.query_and_replace_answer(template)
            return answer if answer else random.choice(APOLOGY_ANSWER)
        else:
            return random.choice(FRESH_ANSWER[fresh_intent])

    def check_slots(self, cypher_slots, slots):
        slots_list = [s[0] for s in slots]
        for slot in cypher_slots:
            if slot not in slots_list:
                return False
        return True

    def replace_token_in_string(self, string, slots):
        for key, value in slots:
            string = string.replace('%' + key + '%', value)
        return string

    def fill_template(self, intent, slots):
        template = TEMPLATES[intent]
        if not self.check_slots(template['slots'], slots):
            return None
        return {
            'cypher': self.replace_token_in_string(template['cypher'], slots),
            'answer': self.replace_token_in_string(template['answer'], slots),
            'confirm': self.replace_token_in_string(template['confirm'], slots),
        }

    def query_and_replace_answer(self, template):
        cypher = template['cypher']
        try:
            result = self.graph.run(cypher).data()
            if result and any(value for value in result[0].values()):
                return self.replace_token_in_string(template['answer'], list(result[0].items()))
        except:
            traceback.print_exc()
            return None

    def parse_diagnose_answer(self, text, uuid):
        score, label_id, label = ir_predict(text)
        if score < THRESHOLD['deny']:
            return random.choice(APOLOGY_ANSWER)

        intent = INTENT_MAP[label_id]
        self.dump_cache({'intent': intent}, uuid)

        slots = ner_predict(text)
        slots_keys = [slot[0] for slot in slots]
        if set(slots_keys) & set(TEMPLATES_SLOTS):
            self.dump_cache({'slots': slots}, uuid)

        cache_data = self.load_cache(uuid)
        template = self.fill_template(cache_data['intent'], cache_data['slots'])
        if not template:
            return None

        if score < THRESHOLD['accept']:
            return template['confirm'] + '（请回答是或者不是）'
        else:
            return self.query_and_replace_answer(template)

    def load_cache(self, uuid):
        try:
            with open(CACHE_PATH % uuid, encoding='utf-8') as file:
                return json.loads(file.read())
        except:
            return {}

    def dump_cache(self, data, uuid):
        result = self.load_cache(uuid)
        result.update(data)
        with open(CACHE_PATH % uuid, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, ensure_ascii=False, indent=2))

    def remove_cache(self, uuid):
        file_path = CACHE_PATH % uuid
        if os.path.exists(file_path):
            os.remove(file_path)

    def generate_llm_supplement(self, question, answer):
        try:
            response = self.llm_client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL"),
                messages=[{
                    "role": "user",
                    "content": (
                        "作为医疗专家，请用30-40个汉字为以下问答提供专业补充：\n"
                        "【强制要求】\n"
                        "1. 必须是语法完整的句子\n"
                        "2. 严格30-40字（宁少勿多）\n"
                        "3. 禁止使用省略号或截断\n"
                        "4. 不要包含'补充'等引导词\n"
                        "5. 使用专业医疗表述\n"
                        "--------------------\n"
                        f"问题：{question}\n"
                        f"回答：{answer}\n"
                        "专业补充："
                    )
                }],
                temperature=0.3,
                max_tokens=50
            )

            supplement = response.choices[0].message.content.strip()

            for prefix in ["补充：", "说明：", "建议：", "注：", "另外", "需要"]:
                supplement = supplement.removeprefix(prefix).strip()

            sentences = [s for s in supplement.split('。') if s]
            if not sentences:
                return ""

            selected = sentences[0] + '。'
            if len(selected) < 30:
                if len(sentences) > 1:
                    selected += sentences[1] + '。'
                else:
                    selected = selected[:-1] + "。" * (30 - len(selected))
            elif len(selected) > 40:
                parts = [p for p in selected.split('，') if p]
                selected = parts[0] + '。' if parts else selected[:40]

            selected = selected[:40]
            return selected if 30 <= len(selected) <= 40 else ""

        except Exception as e:
            print(f"[LLM生成错误] {str(e)}")
            return ""

    def query(self, text, uuid):
        base_answer = self.parse_fresh_answer(text, uuid) or \
                      self.parse_diagnose_answer(text, uuid) or \
                      random.choice(APOLOGY_ANSWER)

        if self.use_llm and base_answer not in APOLOGY_ANSWER:
            supplement = self.generate_llm_supplement(text, base_answer)
            if supplement:
                return base_answer + "\n" + "💡 补充：" + supplement + "\n"

        return base_answer

if __name__ == '__main__':
    qa_system = GraphQA(use_llm=True)

    test_questions = [
        '你好',
        '最近经常头痛是怎么回事？',
        '什么是脑炎?',
        '脑炎还有哪些症状?',
        '去医院看一下的话，应该挂什么科?',
        '不想去医院，可以自己吃药治疗吗？',
        '这病好治吗？',
        '是的'
    ]

    for i, question in enumerate(test_questions, 1):
        answer = qa_system.query(question, uuid=1)
        print(f"Q{i}: {question}")
        print(f"A{i}: {answer}\n{'=' * 50}")