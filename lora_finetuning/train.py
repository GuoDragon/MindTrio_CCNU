# 核心框架
import mindnlp
import mindspore

# 设置NPU上下文
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend", device_id=0)

# 数据处理
from datasets import Dataset
import pandas as pd

# 模型和训练相关
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)

# LoRA相关
from peft import LoraConfig, TaskType, get_peft_model

# 查看版本信息
print(f"mindnlp版本: {mindnlp.__version__}")
print(f"mindspore版本: {mindspore.__version__}")


# 数据路径
train_path = "/home/ma-user/work/data/train.json"
val_path = "/home/ma-user/work/data/val.json"

# 读取数据
df_train = pd.read_json(train_path)
df_val = pd.read_json(val_path)

# 转换为Dataset格式
ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)

# 查看数据集信息
print(f"训练集样本数: {len(ds_train)}")
print(f"验证集样本数: {len(ds_val)}")
print("\n数据集前3个样本:")

# 加载tokenizer
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=False, 
    trust_remote_code=True
)

# 查看tokenizer信息
print(f"词汇表大小: {tokenizer.vocab_size}")
print(f"最大长度: {tokenizer.model_max_length}")
print(f"PAD token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")

# 最大序列长度
MAX_LENGTH = 1024

def process_func(example):
    # 构建指令部分
    instruction = tokenizer(
        f"<|im_start|>system\n你是一位PDTB中文文本关系分析助手<|im_end|>\n"
        f"<|im_start|>user\n{example.get('content', '')}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False
    )

    # 构建回答部分
    response = tokenizer(
        f"{example.get('summary', '')}", 
        add_special_tokens=False
    )

    # 拼接 input_ids 和 attention_mask
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构建 labels：指令部分设为 -100（不计算损失），只对回答部分计算损失
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断到最大长度
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

# 处理训练集和验证集
print("开始处理训练集...")
tokenized_train = ds_train.map(process_func, remove_columns=ds_train.column_names)
print("训练集处理完成！")

print("\n开始处理验证集...")
tokenized_val = ds_val.map(process_func, remove_columns=ds_val.column_names)
print("验证集处理完成！")

# 查看处理后的第一个样本
print("\n处理后的第一个样本解码结果:")
print(tokenizer.decode(tokenized_train[0]['input_ids']))

# 加载基础模型
print("正在加载模型，请稍候...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    ms_dtype=mindspore.bfloat16,
    device_map=0
)

# 显式将模型移动到NPU
model = model.to('npu:0')

# 开启梯度计算
model.enable_input_require_grads()

print("模型加载完成！")
print(f"模型参数量: {model.num_parameters():,}")

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型
    target_modules=[        # 要应用LoRA的模块（注意力层和FFN层）
        "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力层
        "gate_proj", "up_proj", "down_proj"       # FFN层
    ],
    r=16,                   # LoRA秩
    lora_alpha=32,          # LoRA缩放因子
    lora_dropout=0.05,      # Dropout率
    inference_mode=False    # 训练模式
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)

# 打印可训练参数信息
model.print_trainable_parameters()

# 定义训练参数
args = TrainingArguments(
    output_dir="./output",                    # 输出目录
    per_device_train_batch_size=4,            # batch size
    gradient_accumulation_steps=5,            # 梯度累积步数
    logging_steps=10,                         # 日志记录间隔
    num_train_epochs=3,                       # 训练轮数
    save_steps=100,                           # checkpoint保存间隔
    learning_rate=3e-5,                       # 学习率
    save_on_each_node=True,                   # 在每个节点上保存
)

print("训练参数配置完成！")
print(f"有效batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
print(f"总训练步数: {len(tokenized_train) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs}")

# 自定义Trainer类，确保输入数据显式移动到NPU
class NPUTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        """
        将输入数据显式移动到NPU设备
        """
        # 显式将所有输入数据移动到NPU
        inputs = {k: v.to('npu:0') for k, v in inputs.items()}
        return super()._prepare_inputs(inputs)

# 创建Trainer
trainer = NPUTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("Trainer创建成功！")

# 开始训练
print("========== 开始训练 ==========")
print("="*50)
trainer.train()
print("\n========== 训练完成 ==========")

# 查看训练结果文件
import os
print("训练输出目录内容:")
if os.path.exists('./output'):
    for item in sorted(os.listdir('./output')):
        print(f"  - {item}")
else:
    print("输出目录尚未创建")