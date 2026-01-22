import gradio as gr
import mindspore as ms
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os
import sys
import traceback
import time
import torch

# Try to import torch_npu for Ascend NPU support
try:
    import torch_npu
    TORCH_NPU_AVAILABLE = True
    print("[INFO] torch_npu is available")
except ImportError:
    TORCH_NPU_AVAILABLE = False
    print("[WARNING] torch_npu not available, will use CPU")

# Global variables for model and tokenizer
model = None
tokenizer = None

def parse_args():
    parser = argparse.ArgumentParser(description='Gradio App for Sentence Relation Analysis')
    parser.add_argument('--base-model', type=str,
                       default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Base model name or path')
    parser.add_argument('--lora-checkpoint', type=str,
                       default='checkpoint-1380',
                       help='LoRA checkpoint directory')
    parser.add_argument('--device-id', type=int, default=0,
                       help='Device ID (default: 0)')
    return parser.parse_args()

def configure_device(device_id):
    """配置MindSpore设备环境"""
    print(f"\n{'='*60}")
    print(f"配置MindSpore设备环境")
    print(f"{'='*60}")

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",
        device_id=device_id
    )
    print(f"[OK] 设备配置完成 (device_id={device_id})")
    print(f"{'='*60}\n")

def load_model(base_model_name, lora_checkpoint_path):
    """加载基础模型和LoRA适配器"""
    global model, tokenizer

    print("=" * 60)
    print("加载模型")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/4] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
        trust_remote_code=True
    )
    print("      [OK] 分词器加载完成")

    # Load base model
    print("\n[2/4] 加载基础模型...")

    if TORCH_NPU_AVAILABLE:
        print("      使用加速设备...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="npu:0",
            trust_remote_code=True
        )
    else:
        print("      使用CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    print("      [OK] 基础模型加载完成")
    print(f"      模型参数量: {model.num_parameters():,}")

    # Load LoRA adapter
    print("\n[3/4] 加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    print("      [OK] LoRA适配器加载完成")

    # Verify final device
    print("\n[4/4] 检查模型设备...")
    actual_device = next(model.parameters()).device
    print(f"      当前设备: {actual_device}")

    print("\n" + "=" * 60)
    print("模型就绪!")
    print("=" * 60 + "\n")

def analyze_text(user_input):
    """分析中文文本的关系类型"""
    if not user_input or not user_input.strip():
        return "请输入文本"

    try:
        print(f"\n{'='*60}")
        print(f"[推理开始] 输入: {user_input[:50]}...")
        print(f"{'='*60}")

        # 使用chat template格式化输入
        print("[1/4] 格式化输入...")
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "你是PDTB文本关系分析助手"},
                {"role": "user", "content": user_input}
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        print(f"      输入token数: {inputs['input_ids'].shape[1]}")

        # 移动输入到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"      输入已移至: {device}")

        # 生成配置 (根据设备调整)
        print("[2/4] 配置生成参数...")
        device = next(model.parameters()).device

        if device.type == "npu":
            # 使用完整配置
            gen_kwargs = {
                "max_length": 2500,
                "do_sample": True,
                "top_k": 1,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            print(f"      使用完整配置: max_length={gen_kwargs['max_length']}")
        else:
            # 使用快速配置
            gen_kwargs = {
                "max_new_tokens": 100,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            print(f"      使用快速配置: max_new_tokens={gen_kwargs['max_new_tokens']}")
            print(f"      提示: 使用加速设备可获得更好性能")

        # 生成回答
        print("[3/4] 开始生成...")
        start_time = time.time()
        outputs = model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start_time
        print(f"      生成完成，耗时: {elapsed:.2f}秒")

        # 只保留生成的部分（去除输入）
        print("[4/4] 解码输出...")
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 截取 </think> 之后的内容
        think_end = response.find("</think>")
        if think_end != -1:
            response = response[think_end + len("</think>"):].strip()

        # 清理剩余的特殊标记
        response = response.replace("<｜end▁of▁sentence｜>", "").strip()

        print(f"[推理完成] 输出长度: {len(response)}")
        print(f"{'='*60}\n")

        return response if response else "模型未返回有效结果"

    except Exception as e:
        # 详细错误日志
        error_msg = f"分析出错: {str(e)}"
        print(f"\n{'!'*60}")
        print(f"[错误] {error_msg}")
        print(f"{'!'*60}")
        print("详细错误信息:")
        traceback.print_exc(file=sys.stdout)
        print(f"{'!'*60}\n")
        return error_msg

def main():
    args = parse_args()

    # Get paths
    base_model = args.base_model
    lora_checkpoint = args.lora_checkpoint

    print(f"Base model: {base_model}")
    print(f"LoRA checkpoint: {lora_checkpoint}")

    # Configure device
    configure_device(args.device_id)

    # Load model
    load_model(base_model, lora_checkpoint)

    # Create Gradio interface
    with gr.Blocks(title="中文句子关系分析系统") as demo:
        gr.Markdown("# 中文句子关系分析系统")
        gr.Markdown("输入中文句子，分析其中的关系类型（扩展、并列、因果、转折、其他）")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="输入文本",
                    placeholder="请输入要分析的中文句子...",
                    lines=3
                )
                analyze_btn = gr.Button("分析", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="分析结果",
                    lines=5
                )

        gr.Markdown("### 示例")
        gr.Examples(
            examples=[
                ["第一个问题是什么问题？诗中哪个字统领全篇？"],
                ["海鸟是一个胆怯的形象，它想干嘛？写这些其他的海鸟是为了干什么呢？"],
                ["月亮又圆又亮,所以古人称之为玉盘。"]
            ],
            inputs=input_text
        )

        analyze_btn.click(
            fn=analyze_text,
            inputs=input_text,
            outputs=output_text
        )

    # Launch
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
