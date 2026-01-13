# 核心框架
import mindnlp
import mindspore

# 设置NPU上下文
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend", device_id=0)

# 模型相关
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 查看版本信息
print(f"mindnlp版本: {mindnlp.__version__}")
print(f"mindspore版本: {mindspore.__version__}")

# 基础模型名称
base_model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

# 训练好的 LoRA checkpoint 路径
lora_path = "/home/ma-user/work/output/checkpoint-1380"

# 合并后模型的保存目录
merged_path = "/home/ma-user/work/merged_model"

print(f"基础模型: {base_model_name}")
print(f"LoRA权重路径: {lora_path}")
print(f"合并后保存路径: {merged_path}")

# 加载tokenizer
print("正在加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, 
    use_fast=False, 
    trust_remote_code=True
)
print("Tokenizer加载完成！")

# 加载基础模型
print("正在加载基础模型，请稍候...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    ms_dtype=mindspore.bfloat16,  # 使用bfloat16数据类型
    device_map=0  # 指定设备
)

print("基础模型加载完成！")
print(f"模型参数量: {model.num_parameters():,}")

# 加载 LoRA 适配器权重
print("正在加载LoRA适配器权重...")
model = PeftModel.from_pretrained(model, lora_path)
print("LoRA权重加载完成！")

# 合并 LoRA 权重到基础模型
print("正在合并权重...")
model = model.merge_and_unload()
print("权重合并完成！")

# 保存完整的微调模型
print(f"正在保存模型到 {merged_path}...")
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

print()
print("LoRA 权重已成功合并！")
print(f"合并后的模型保存在: {merged_path}")

# 将模型移至NPU设备
print("正在将模型移至NPU设备...")
model = model.to('npu:0')
print("模型已就绪!")

# 测试样例
test_prompt = "月亮又圆又亮,所以古人称之为玉盘。"

print("="*60)
print("推理测试")
print("="*60)
print(f"输入文本: {test_prompt}")
print("-"*60)

# 构建对话输入
inputs = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "你是PDTB文本关系分析助手"},
        {"role": "user", "content": test_prompt}
    ],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="ms",
    return_dict=True
)

# 显式将所有输入数据移动到NPU
inputs = {k: v.to('npu:0') for k, v in inputs.items()}

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