#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDTB中文文本关系分析模型训练脚本
使用MindSpore + LoRA在DeepSeek-R1-Distill-Qwen-1.5B模型上进行微调
"""

import argparse
import mindnlp
import mindspore
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PDTB中文文本关系分析模型训练脚本')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, 
                        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                        help='预训练模型名称')
    
    # 数据参数
    parser.add_argument('--train_path', type=str, 
                        default='./data/train.json',
                        help='训练集路径')
    parser.add_argument('--val_path', type=str, 
                        default='./data/val.json',
                        help='验证集路径')
    parser.add_argument('--max_length', type=int, 
                        default=1024,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str, 
                        default='./output',
                        help='输出目录')
    parser.add_argument('--per_device_train_batch_size', type=int, 
                        default=4,
                        help='每个设备的训练batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, 
                        default=5,
                        help='梯度累积步数')
    parser.add_argument('--logging_steps', type=int, 
                        default=10,
                        help='日志记录间隔')
    parser.add_argument('--num_train_epochs', type=int, 
                        default=3,
                        help='训练轮数')
    parser.add_argument('--save_steps', type=int, 
                        default=100,
                        help='checkpoint保存间隔')
    parser.add_argument('--learning_rate', type=float, 
                        default=3e-5,
                        help='学习率')
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int, 
                        default=16,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, 
                        default=32,
                        help='LoRA缩放因子')
    parser.add_argument('--lora_dropout', type=float, 
                        default=0.05,
                        help='LoRA dropout率')
    
    # 设备参数
    parser.add_argument('--device_target', type=str, 
                        default='Ascend',
                        choices=['Ascend', 'GPU', 'CPU'],
                        help='设备类型')
    parser.add_argument('--device_id', type=int, 
                        default=0,
                        help='设备ID')
    
    return parser.parse_args()

def process_func(example, tokenizer, max_length):
    """数据预处理函数
    
    Args:
        example: 单个样本数据
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        处理后的样本数据
    """
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
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备上下文
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, 
                        device_target=args.device_target, 
                        device_id=args.device_id)
    
    # 查看版本信息
    print(f"mindnlp版本: {mindnlp.__version__}")
    print(f"mindspore版本: {mindspore.__version__}")
    
    # 读取数据集
    print(f"\n正在加载数据集...")
    df_train = pd.read_json(args.train_path)
    df_val = pd.read_json(args.val_path)
    
    # 转换为Dataset格式
    ds_train = Dataset.from_pandas(df_train)
    ds_val = Dataset.from_pandas(df_val)
    
    # 查看数据集信息
    print(f"训练集样本数: {len(ds_train)}")
    print(f"验证集样本数: {len(ds_val)}")
    print("数据集前3个样本:")
    print(ds_train[:3])
    
    # 加载tokenizer
    print(f"\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        use_fast=False, 
        trust_remote_code=True
    )
    
    # 查看tokenizer信息
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"最大长度: {tokenizer.model_max_length}")
    print(f"PAD token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    
    # 数据预处理
    print(f"\n正在处理数据集...")
    print("开始处理训练集...")
    tokenized_train = ds_train.map(
        lambda example: process_func(example, tokenizer, args.max_length), 
        remove_columns=ds_train.column_names
    )
    print("训练集处理完成！")
    
    print("\n开始处理验证集...")
    tokenized_val = ds_val.map(
        lambda example: process_func(example, tokenizer, args.max_length), 
        remove_columns=ds_val.column_names
    )
    print("验证集处理完成！")
    
    # 查看处理后的第一个样本
    print("\n处理后的第一个样本解码结果:")
    print(tokenizer.decode(tokenized_train[0]['input_ids']))
    
    # 加载基础模型
    print(f"\n正在加载模型，请稍候...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        ms_dtype=mindspore.bfloat16,  # 使用bfloat16数据类型
        device_map=args.device_id     # 指定设备
    )
    
    # 显式将模型移动到设备
    model = model.to(f'{args.device_target.lower()}:{args.device_id}')
    
    # 开启梯度计算
    model.enable_input_require_grads()
    
    print("模型加载完成！")
    print(f"模型参数量: {model.num_parameters():,}")
    
    # 配置LoRA
    print(f"\n正在配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型
        target_modules=[        # 要应用LoRA的模块（注意力层和FFN层）
            "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力层
            "gate_proj", "up_proj", "down_proj"       # FFN层
        ],
        r=args.lora_r,                   # LoRA秩
        lora_alpha=args.lora_alpha,      # LoRA缩放因子
        lora_dropout=args.lora_dropout,  # Dropout率
        inference_mode=False             # 训练模式
    )
    
    # 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    # 定义训练参数
    print(f"\n正在配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,                    # 输出目录
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积步数
        logging_steps=args.logging_steps,              # 日志记录间隔
        num_train_epochs=args.num_train_epochs,        # 训练轮数
        save_steps=args.save_steps,                    # checkpoint保存间隔
        learning_rate=args.learning_rate,              # 学习率
        save_on_each_node=True,                        # 在每个节点上保存
    )
    
    print("训练参数配置完成！")
    print(f"有效batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"总训练步数: {len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    # 创建Trainer
    print(f"\n正在创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    print("Trainer创建成功！")
    
    # 开始训练
    print(f"\n========== 开始训练 ==========")
    trainer.train()
    print("\n========== 训练完成 ==========")

if __name__ == "__main__":
    main()