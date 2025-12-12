#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA权重合并脚本
将训练好的LoRA权重合并到基础模型中
"""

import argparse
import mindnlp
import mindspore
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LoRA权重合并脚本')
    
    # 核心参数
    parser.add_argument('--base_model', type=str, 
                        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                        help='基础模型名称')
    parser.add_argument('--lora_path', type=str, 
                        default='./checkpoint-1380',
                        help='训练好的LoRA权重路径')
    parser.add_argument('--merged_path', type=str, 
                        default='./merged_model',
                        help='合并后模型的保存路径')
    
    # 设备参数
    parser.add_argument('--device_target', type=str, 
                        default='Ascend',
                        choices=['Ascend', 'GPU', 'CPU'],
                        help='设备类型')
    parser.add_argument('--device_id', type=int, 
                        default=0,
                        help='设备ID')
    
    # 推理测试参数
    parser.add_argument('--test_prompt', type=str, 
                        default='月亮又圆又亮,所以古人称之为玉盘。',
                        help='推理测试的输入文本')
    
    return parser.parse_args()

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
    
    # 打印配置信息
    print(f"\n基础模型: {args.base_model}")
    print(f"LoRA权重路径: {args.lora_path}")
    print(f"合并后保存路径: {args.merged_path}")
    
    # 加载tokenizer
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        use_fast=False, 
        trust_remote_code=True
    )
    print("Tokenizer加载完成！")
    
    # 加载基础模型
    print("\n正在加载基础模型，请稍候...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        ms_dtype=mindspore.bfloat16,  # 使用bfloat16数据类型
        device_map=args.device_id     # 指定设备
    )
    
    print("基础模型加载完成！")
    print(f"模型参数量: {model.num_parameters():,}")
    
    # 加载LoRA适配器权重
    print("\n正在加载LoRA适配器权重...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("LoRA权重加载完成！")
    
    # 合并LoRA权重到基础模型
    print("\n正在合并权重...")
    model = model.merge_and_unload()
    print("权重合并完成！")
    
    # 保存完整的微调模型
    print(f"\n正在保存模型到 {args.merged_path}...")
    model.save_pretrained(args.merged_path)
    tokenizer.save_pretrained(args.merged_path)
    
    print("\nLoRA 权重已成功合并！")
    print(f"合并后的模型保存在: {args.merged_path}")
    
    # 推理测试
    print("\n" + "="*60)
    print("推理测试")
    print("="*60)
    print(f"输入文本: {args.test_prompt}")
    print("-"*60)
    
    # 将模型移至设备
    print("正在将模型移至设备...")
    model = model.to(f'{args.device_target.lower()}:{args.device_id}')
    print("模型已就绪!")
    
    # 构建对话输入
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是PDTB文本关系分析助手"},
            {"role": "user", "content": args.test_prompt}
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="ms",
        return_dict=True
    )
    
    # 显式将所有输入数据移动到设备
    inputs = {k: v.to(f'{args.device_target.lower()}:{args.device_id}') for k, v in inputs.items()}
    
    # 生成配置
    gen_kwargs = {
        "max_length": 2500,
        "do_sample": True,
        "top_k": 1
    }
    
    # 生成回答
    outputs = model.generate(**inputs, **gen_kwargs)
    # 只保留生成的部分(去除输入)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 截取 </think> 之后的内容
    think_end = response.find("</think>")
    if think_end != -1:
        response = response[think_end + len("</think>"):].strip()
    
    print("模型输出:")
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()