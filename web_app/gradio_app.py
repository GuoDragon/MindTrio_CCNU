import gradio as gr
import mindspore as ms
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

# Global variables for model and tokenizer
model = None
tokenizer = None

def parse_args():
    parser = argparse.ArgumentParser(description='Gradio App for Sentence Relation Analysis (NPU Version)')
    parser.add_argument('--base-model', type=str,
                       default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                       help='Base model name or path')
    parser.add_argument('--lora-checkpoint', type=str,
                       default='checkpoint-1380',
                       help='LoRA checkpoint directory')
    parser.add_argument('--device-id', type=int, default=0,
                       help='NPU device ID (default: 0)')
    return parser.parse_args()

def configure_device(device_id):
    """Configure MindSpore Ascend NPU context (与华为云一致)"""
    print(f"\n{'='*60}")
    print(f"Configuring MindSpore Ascend NPU")
    print(f"{'='*60}")

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",  # 华为云Ascend NPU
        device_id=device_id
    )
    print(f"[OK] Ascend NPU configured (device_id={device_id})")
    print(f"{'='*60}\n")

def load_model(base_model_name, lora_checkpoint_path):
    """Load base model + LoRA adapter (与华为云test文件夹相同)"""
    global model, tokenizer

    print("=" * 60)
    print("Loading Model (PyTorch Transformers + PEFT)")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
        trust_remote_code=True
    )
    print("      [OK] Tokenizer loaded")

    # Load base model
    print("\n[2/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        ms_dtype=ms.bfloat16,  # 使用bfloat16（与华为云一致）
        device_map=0,
        trust_remote_code=True
    )
    print("      [OK] Base model loaded")
    print(f"      Model parameters: {model.num_parameters():,}")

    # Load LoRA adapter
    print("\n[3/4] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)
    print("      [OK] LoRA adapter loaded")

    # Move to NPU
    print("\n[4/4] Moving model to NPU...")
    model = model.to('npu:0')
    print("      [OK] Model ready on NPU")

    print("\n" + "=" * 60)
    print("Model Ready!")
    print("=" * 60 + "\n")

def analyze_text(user_input):
    """Analyze Chinese text for relation classification (使用chat template)"""
    if not user_input or not user_input.strip():
        return "请输入文本"

    try:
        # 使用chat template格式化输入
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "你是PDTB文本关系分析助手"},
                {"role": "user", "content": user_input}
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="ms",
            return_dict=True
        )

        # 将输入移至NPU
        inputs = {k: v.to('npu:0') for k, v in inputs.items()}

        # 生成配置
        gen_kwargs = {
            "max_length": 2500,
            "do_sample": True,
            "top_k": 1
        }

        # 生成回答
        outputs = model.generate(**inputs, **gen_kwargs)

        # 只保留生成的部分（去除输入）
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 截取 </think> 之后的内容
        think_end = response.find("</think>")
        if think_end != -1:
            response = response[think_end + len("</think>"):].strip()

        return response if response else "模型未返回有效结果"

    except Exception as e:
        return f"分析出错: {str(e)}"

def main():
    args = parse_args()

    # Get paths
    base_model = args.base_model
    lora_checkpoint = args.lora_checkpoint

    print(f"Base model: {base_model}")
    print(f"LoRA checkpoint: {lora_checkpoint}")

    # Configure NPU device
    configure_device(args.device_id)

    # Load model
    load_model(base_model, lora_checkpoint)

    # Create Gradio interface
    with gr.Blocks(title="中文句子关系分析 (NPU版本)") as demo:
        gr.Markdown("# 中文句子关系分析系统 (NPU版本)")
        gr.Markdown("**配置**: PyTorch Transformers + PEFT + MindSpore Ascend NPU")
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
