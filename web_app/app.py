from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import json
from tabulate import tabulate
import os
import uuid
import re
import mindspore as ms
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
import random
import argparse
import sys

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Chinese Sentence Relation Analysis Web App')
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the model directory (default: from env MODEL_PATH or D:\\hw\\merge\\merged_model)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run Flask app on (default: 5000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind Flask app to (default: 127.0.0.1)'
    )
    return parser.parse_args()

app = Flask(__name__)

# 数据库配置
DB_PATH = 'relation_analysis.db'  # SQLite 数据库文件路径

# 解析命令行参数
args = parse_args()

# 本地模型配置（优先级：命令行参数 > 环境变量 > 默认值）
MODEL_ID = (
    args.model_path or
    os.environ.get('MODEL_PATH') or
    r"D:\hw\merge\merged_model"
)

print(f"使用模型路径: {MODEL_ID}")

# Few-shot 示例（优化为2个示例以提升推理速度）
FEW_SHOT_EXAMPLES = """
    1、输入：第一个问题是什么问题？诗中哪个字统领全篇？
    输出：扩展。原因：前半句提出问题，后半句进行了扩展。

    2、输入：海鸟是一个胆怯的形象，它想干嘛？写这些其他的海鸟是为了干什么呢？
    输出：并列。原因：前后两部分分别针对不同方面提问，相互平行独立。
"""

# 模型加载 - 在脚本启动时加载模型，只加载一次
print("=" * 60)
print(f"正在从 {MODEL_ID} 加载模型...")
print("=" * 60)
try:
    # 加载tokenizer
    print("[1/3] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    print("     [OK] Tokenizer加载成功")

    # 加载模型
    print("[2/3] 加载模型（这可能需要几分钟）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        ms_dtype=ms.float16,
        trust_remote_code=True
    )
    print("     [OK] 模型加载成功")

    # 设置为推理模式
    print("[3/3] 设置推理模式...")
    model.set_train(False)
    print("     [OK] 推理模式设置完成")

    # 修复pad_token问题
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print("     [修复] pad_token与eos_token相同，正在修复...")
        # 尝试使用unk_token作为pad_token
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            print(f"     [OK] 使用unk_token作为pad_token: {tokenizer.unk_token_id}")
        else:
            # 如果没有unk_token，使用词表中的第一个token
            tokenizer.pad_token_id = 0
            print("     [OK] 使用token_id=0作为pad_token")

    print("=" * 60)
    print("[SUCCESS] 模型加载完成！应用已准备就绪。")
    print("=" * 60)
except Exception as e:
    print("=" * 60)
    print(f"[ERROR] 模型加载失败: {e}")
    print("请确保模型文件已正确下载到指定路径。")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    model = None
    tokenizer = None

FIVE_CLASSES = ["并列", "扩展", "比较", "因果", "其他"]  # 只有五种分类


def split_long_text(text, max_len=70):
    """
    将长文本按标点优先切分，每段不超过 max_len ，句子本身超长则强制截断。
    """
    result = []  # 初始化结果列表
    sentences = re.split(r'(?<=[。！？!?.])', text)  # 通过正则根据中英文标点划分句子
    buf = ''  # 当前句子的内容

    # 遍历考虑划分后的每一个句子
    for sent in sentences:
        # 如果 sent 去掉空格换行后为空，直接跳过
        if not sent.strip():
            continue

        # 当前句子加上考虑句后长度仍然小于等于 max_len
        if len(buf) + len(sent) <= max_len:
            buf += sent
        else:   # 长度和大于 max_len，两句话不能合在一起
            if buf:     # 当前句子不为空，就直接作为一个结果加入到 result 列表中
                result.append(buf)

            # 待考虑句本身长度小于 max_len，就将他作为新的当前句
            if len(sent) <= max_len:
                buf = sent
            else:
                # 句子本身超长，强制截断
                for i in range(0, len(sent), max_len):
                    part = sent[i:i+max_len]
                    result.append(part)
                buf = ''

    # 最后只剩下当前句，而没有新的待考虑句
    if buf:
        result.append(buf)

    return result


def extract_dialogue_start(text):
    """
    跳过开头说明、表格等内容，从第一个T:或S:开头的地方开始提取，支持全角/半角冒号
    """
    match = re.search(r'(^|\n)[ \t]*(T|S)[:：]', text)
    if match:
        return text[match.start():]
    else:
        return text  # 如果没有T:或S:，则不处理


def parse_dialogue(text):
    """
    1. 默认 T 和 S 配对一组。
    2. T 或 S 超过 70 字则按标点优先切分为多个 ≤ 70 字小段。
    3. T 或 S 小于 10 字时，复制上个人最后一句合并（原分段不变，合成新分组）。
    4. 没有 T/S 标记时，按段落切分。
    """
    # 提取每个人说的完整的话
    pattern = r'(T[:：].*?(?=S[:：]|T[:：]|$))|(S[:：].*?(?=T[:：]|S[:：]|$))'

    matches = re.finditer(pattern, text, re.DOTALL)
    segments = []  # 存储每个人的完整的话
    for m in matches:
        seg = m.group(0).strip()
        if seg:
            segments.append(seg)

    # 没有 T/S 标记时，按段落切分（实现第4条功能）
    if not segments:
        # 按换行符划分段落
        paragraphs = text.split('\n')
        # 过滤空段落并保留有内容的
        segments = [p.strip() for p in paragraphs if p.strip()]

    # 返回上一段话的最后一个句子
    def get_last_sentence(paragraph):
        sentences = re.split(r'(?<=[。！？!?.])', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else paragraph

    turns = []          # 记录最后的分句情况
    i = 0               # 记录当前是第几个人在说话
    last_full = None    # 记录上个人的话

    # 遍历每一个人说的完整的话
    while i < len(segments):
        seg = segments[i]   # seg 记录当前句
        person = seg[:2] if seg[:2] in ('T:', 'S:') else None   # 记录当前句是谁说的

        # 超过 70 字，按标点优先切分
        if len(seg) > 70:
            for part in split_long_text(seg, 70):
                turns.append(part)
            last_full = seg
            i += 1
            continue

        # 小于 10 字，复制上个人最后一句合并（原分段不变，合成新分组）
        if len(seg) < 10 and last_full is not None:
            last_sentence = get_last_sentence(last_full)
            turns.append(f"{last_sentence}{seg}")
            i += 1
            continue

        # 尝试 T + S 配对
        if person == 'T:' and i+1 < len(segments) and segments[i+1].startswith('S:'):
            next_seg = segments[i+1]

            # T + S 超过 70 字，只将 T 加入到最后结果中
            if len(next_seg) + len(seg) > 70:
                turns.append(seg)
                last_full = seg
                i += 1
                continue

            # 正常 T + S 配对
            turns.append(seg + next_seg)
            last_full = next_seg
            i += 2
        else:
            turns.append(seg)
            last_full = seg
            i += 1

    return turns


def parse_model_output(text):
    """
    解析模型输出，提取关系类型和原因。优先严格匹配，其次模糊推断，最后随机兜底。
    """
    FIVE_CLASSES = ["并列", "扩展", "比较", "因果", "其他"]

    # 优先尝试严格格式匹配
    match = re.search(r"关系类型[:：]?\s*(.*?)\s*\n\s*原因[:：]?\s*(.*)", text, re.DOTALL)
    if match:
        classification = match.group(1).strip()  # 提取分类
        reason = match.group(2).strip()          # 提取原因
    else:
        # 模型输出不规范，先提取"原因"
        reason_match = re.search(r"原因[:：]?\s*(.*)", text, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else None

        classification = None
        if reason:
            # 按中文标点句号、问号、感叹号分句
            sentences = re.split(r'(?<=[。！？!?])', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # 先找包含"原因"的句子索引
            reason_idx = None
            for idx, sent in enumerate(sentences):
                if "原因：" in sent or "原因:" in sent:
                    reason_idx = idx
                    break

            # 再往前找包含分类标签的句子
            if reason_idx is not None:
                for prev_sent in reversed(sentences[:reason_idx]):
                    for label in FIVE_CLASSES:
                        if label in prev_sent:
                            classification = label
                            break
                    if classification:
                        break

        # 兜底随机一个
        if classification not in FIVE_CLASSES:
            classification = random.choice(FIVE_CLASSES)
            if not reason:
                reason = f"模型未能明确判断，随机归为\"{classification}\"关系。"

    # 如果识别出分类但没有原因，给默认理由
    if not reason:
        reason = f"模型未能给出详细理由。"

    return classification, reason


def call_model(user_input):
    """
    调用模型分析单个文本单元，并返回符合格式的结果。
    """
    if model is None or tokenizer is None:
        return {"sentence": user_input, "classification": "错误", "reason": "模型未成功加载，请检查模型路径是否正确。"}

    print(f"--- 分析单元: {user_input[:20]}... ---")    # 打印日志

    # 构建 Prompt
    prompt = f"""
    你是一个中文句间关系逻辑分析专家，请判断下列句子的句间逻辑关系，并根据具体的句子内容解释你的判断理由。
    请参考以下示例并严格按照格式回答，注意将思考过程用  和  标签明确区分于最终结果：

    {FEW_SHOT_EXAMPLES}

    【待判断】
    句子：{user_input}
    请严格按照以下格式作答：
    关系类型：（并列 / 扩展 / 比较 / 因果 / 其他）
    原因：（详细说明你的判断理由）
    """

    messages = [
        {"role": "system", "content": "你是一个擅长分析句间逻辑关系的中文专家，请根据句子具体内容输出关系类型和原因，注意将思考过程用  和  标签明确区分于最终结果。"},
        {"role": "user", "content": prompt}
    ]

    # 构建 chat prompt
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,             # 返回字符串，而不是直接返回 token id
        add_generation_prompt=True  # 在文本末尾加上提示生成的特殊标记
    )

    # 设置模型生成文本时的终止符
    terminators = [tokenizer.eos_token_id]  # 默认终止符
    try:
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # 自定义终止符
        if eot_token_id is not None and eot_token_id != tokenizer.unk_token_id:
            terminators.append(eot_token_id)  # 遇到其中一个终止符就结束生成
    except Exception:
        pass  # EOT token不存在时忽略

    # 模型生成
    print("    [推理] 开始tokenize...")
    # 1. Tokenize输入
    inputs = tokenizer(chat_prompt, return_tensors="ms")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)  # 获取attention_mask
    print(f"    [推理] Tokenize完成，输入长度: {input_ids.shape[1]} tokens")

    # 2. 模型生成
    print("    [推理] 开始模型推理（可能需要几秒到几分钟）...")
    import time
    start_time = time.time()

    # 构建generate参数
    generate_kwargs = {
        "max_new_tokens": 128,  # 进一步降低到128
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,     # 使用贪心搜索，速度更快
        "temperature": 0.7,
        "top_p": 0.9
    }

    # 如果有attention_mask，则传入
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
        print("    [推理] 使用attention_mask")
    else:
        print("    [推理] 警告：没有attention_mask")

    outputs = model.generate(input_ids, **generate_kwargs)

    elapsed_time = time.time() - start_time
    print(f"    [推理] 推理完成，耗时: {elapsed_time:.2f}秒")

    # 3. 解码输出（跳过输入部分）
    print("    [推理] 解码输出...")
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # 截取 之后的内容
    think_end = generated_text.find("")
    if think_end != -1:
        cleaned_output = generated_text[think_end + len(""):].strip()
    else:
        cleaned_output = generated_text.strip()
    print(f"模型原始输出:\n{generated_text}")
    print(f"清洗输出:\n{cleaned_output}")

    classification, reason = parse_model_output(cleaned_output)

    result = {
        "sentence": user_input,
        "classification": classification,
        "reason": reason
    }

    print(f"解析结果: 类型='{classification}', 原因='{reason[:20]}...'")

    return result


# 通过配置参数连接本地 SQLite 数据库，并返回连接对象
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
    return conn


def format_db_results(results):
    """
    将原始数据库结果整理成一个 更容易 JSON 化的字典列表格式
    results ：数据库查询后返回的结果列表
    """
    if not results:
        return []

    # 格式化每一行数据
    formatted_results = []  # 保存格式化后的字典结果
    for row in results:  # 遍历每一行数据库返回的数据
        formatted_row = {}
        for key in row.keys():
            # 处理日期时间格式
            if isinstance(row[key], datetime):
                formatted_row[key] = row[key].strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted_row[key] = row[key]
        formatted_results.append(formatted_row)

    return formatted_results


@app.route('/')  # 当浏览器访问根路径 http://localhost:5000/ 时触发该函数
def index():    # 加载并渲染 templates/ 文件夹下的 index.html 页面
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])    # 后端分析接口
def analyze():
    '''
    1、 将用户输入进行划分
    2、 调用本地模型进行分析，生成输出
    '''
    # 从前端发来的 JSON 请求体中提取文本
    text = request.json['text']
    text = extract_dialogue_start(text)  # 新增：去除开头说明
    print("收到文本：", text)

    dialogue_turns = parse_dialogue(text)
    print("分组结果：", dialogue_turns)

    if not dialogue_turns:
        return jsonify([])

    print(f"解析出 {len(dialogue_turns)} 个对话轮次进行分析。")

    all_results = []

    for i, task_text in enumerate(dialogue_turns):
        print(f"--- 正在处理子任务 {i+1}/{len(dialogue_turns)} ---")
        result = call_model(task_text)
        all_results.append(result)

    # 直接将结果返回给前端
    return jsonify(all_results)


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    接收带有用户判断的完整分析结果，并将其存入数据库
    """
    data = request.json         # 期望接收包含 {sentence, classification, reason, is_correct} 的对象
    conn = get_db_connection()  # 连接到 SQLite 数据库，返回一个连接对象 conn
    cursor = conn.cursor()      # 通过 conn 创建一个数据库游标，用于执行 SQL 语句

    now = datetime.now()        # 当前数据的存储时间

    # 在用户反馈时，将完整记录一次性写入数据库
    cursor.execute('''
        INSERT INTO analysis_results
        (sentence, classification, reason, is_correct, created_at, feedback_time)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['sentence'], data['classification'], data['reason'],
         data['is_correct'], now, now))

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'status': 'success', 'message': '数据已成功存入数据库'})


# 如果没有，先创建数据库的表
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            classification VARCHAR(50) NOT NULL,
            reason TEXT NOT NULL,
            is_correct INTEGER,
            created_at DATETIME NOT NULL,
            feedback_time DATETIME
        )
    ''')

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    init_db()
    print(f"启动Flask应用于 http://{args.host}:{args.port}")
    app.run(debug=False, host=args.host, port=args.port)  # Temporarily disable debug mode
