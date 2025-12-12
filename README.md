# 面向课堂场景的大模型定制化微调：逻辑问答交互与授课效果的关联性研究

## 项目背景和简介
在传统课堂话语分析体系中，使用特征工程的方法虽然效果不错，但需要耗费大量的人力物力；另一种使用 Transformer 等经典深度学习的方法虽然非常高效，但是典型的“黑盒”模型，可解释性弱，而在实际课堂分析中，教师往往看重可解释性。因此，本项目实现了：
- 通过使用微调大模型，在保证性能的同时，兼顾可解释性
- 提供相应的封装代码，实现端到端的程序调用

## 技术方案
本项目基于CDTB数据集，采用MindSpore框架微调DeepSeek-R1-Distill-Qwen-1.5B大模型，实现了中文篇章级句间关系识别任务。

### 技术栈与核心组件
| 技术栈           | 版本/说明                         |
| ---------------- | --------------------------------- |
| 深度学习框架     | MindSpore                         |
| NLP工具库        | mindnlp                           |
| 大语言模型       | DeepSeek-R1-Distill-Qwen-1.5B     |
| 参数高效微调方法 | LoRA (Low-Rank Adaptation)        |
| 数据集           | CDTB (Chinese Discourse TreeBank) |

### 核心技术实现细节

#### 数据处理流程
```python
# 数据路径
train_path = "/home/ma-user/work/data/train.json"
val_path = "/home/ma-user/work/data/val.json"

# 读取数据
df_train = pd.read_json(train_path)
df_val = pd.read_json(val_path)

# 转换为Dataset格式
ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)

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
```

#### LoRA微调配置
```python
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
```

#### 模型训练与合并
```python
# 训练配置
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

# 训练执行
# 创建Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()

# LoRA权重合并
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(merged_path)
```


## 数据集准备
### 原始数据
本项目使用哈尔滨工业大学 CDTB 数据集，并在此基础上进行了一定的补充，保障了数据的规模、多样性和高质量。各位开发者可通过`https://ir.hit.edu.cn/2024/1029/c19699a357757/page.htm`向哈工大官方网站申请调用。

### 数据集介绍
哈工大 CDTB 即 HIT - CDTB（哈尔滨工业大学中文篇章关系语料），核心借鉴了 PDTB 的标注标准与研究框架，将篇章级中文语义共分为：
- 扩展
- 因果
- 比较
- 并列
- 时序
- 条件

其中，由于**时序**和**条件**两类的样本数量过少，本实验中将其统一为**其他**关系

### 数据集处理
为保证实验效果，我们在 CDTB 数据集的基础上，做出了一定的修改：
1. 原始样本格式为：
    ```json
    {
        "content": "星汉是什么？银河。",
        "summary": "扩展"
    }
    ```
    为解决可解释性问题，我们决定扩充数据集，为数据加上具体的原因类标签而非只是分类结果。扩充后的样本格式为：
    ```json
    {
        "content": "他的有没有什么不足之处？我觉得他可以就是加一些他自己的感受，因为他如果光只说那些一系列的动作，就感觉很空白，没有什么情感在里面。",
        "summary": "扩展\n原因：前半句话提出问题，询问他的不足之处，后半句话则具体回答了我认为的他的不足之处，所以属于扩展关系。"
    }
    ```
2. 由于时间有限，我们只从一万条数据中随机抽取了两千条数据，并为他们手动打上了原因。但为保持微调效果，我们为其余数据统一添加了“原因：”字样，即：
   ```json
    {
        "content": "星汉是什么？银河。",
        "summary": "扩展\n原因："
    }
   ```
3. 训练集和测试集的比例按 8 : 2 的比例在保证分类比例不变的情况下随机划分


## 项目使用
本项目微调部分在华为云上实现，页面部分在本地 Windows 系统上实现
### 环境搭建
1. 本项目使用虚拟环境`python3 -m venv lcl`安装相应环境，也可以直接使用全局环境
2. 通过`source /home/ma-user/work/lcl/bin/activate`激活虚拟环境
3. **安装MindNLP**
   ```bash
   pip install mindnlp==0.5.1
   ```
4. 安装 4.5.1 版本的 MinsNLP，会自动安装所需的其他依赖库，如 MindSpore、Transformers 等
5. 下载 ipykernel `pip install ipykernel`，用于在 Jupyter Notebook 中运行 Python 代码
6. 注册当前虚拟环境的 Jupyter 内核 `python -m ipykernel install --user --name lcl --display-name "Python (lcl)"`
7. 在 Jupyter Notebook 中选择 `Kernel -> Change kernel -> Python (lcl)` 即可切换到当前虚拟环境
8. 注意每次重启 Notebook 后，都需要重新执行步骤 6, 7

### 数据准备
1. 准备 CDTB 格式的 JSON 数据集文件
2. 数据集格式要求：
   ```json
   {
     "content": "文本内容",
     "summary": "关系类型\n原因：关系解释"
   }
   ```
3. 将数据集划分为train.json和val.json，放置于data目录下

### 模型训练
使用`train.ipynb`进行模型训练：
- 训练过程将生成output目录，保存checkpoint文件
- 可通过日志查看loss、epoch等训练指标

### LoRA权重合并
使用`merge.ipynb`合并LoRA权重与基础模型

### 本地环境搭建
1. 为便于环境搭建，推荐使用 anaconda 单独创建虚拟环境`conda create -n mindtrio python=3.10`
2. 安装 GPU 版本 Pytorch `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` ，或 CPU 版本 `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
3. 安装 `requirements.txt` 中的依赖包 `pip install -r requirements.txt`

### 使用脚本
1. 提前将微调合并后的模型下载下来，并将 `app.py` 中第 25 行的 `MODEL_ID` 变量修改为实际模型路径
2. 在虚拟环境中运行 `python app.py`，预计输出
    ```bash
    正在从 YOUR_PATH 加载模型...
    Device set to use cuda:0
    模型加载完成。
    * Serving Flask app 'app'
    * Debug mode: on
    WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    * Running on http://127.0.0.1:5000
    Press CTRL+C to quit
    * Restarting with stat
    正在从 YOUR_PATH 加载模型...
    Device set to use cuda:0
    模型加载完成。
    * Debugger is active!
    * Debugger PIN: 113-062-594
    ```
3. 在浏览器中打开 `http://127.0.0.1:5000` 后，就可以进行系统的使用