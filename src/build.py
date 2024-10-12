import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的 GPT2 模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 添加填充标记
tokenizer.pad_token = tokenizer.eos_token

# 加载数据集，这里使用了 Hugging Face 的数据集库
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 数据预处理：分词、编码并限制序列长度
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # 模型输出目录
    evaluation_strategy="epoch",  # 评估策略，每个 epoch 进行一次评估
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=4,  # 每个设备的训练批次大小
    per_device_eval_batch_size=4,  # 每个设备的评估批次大小
    num_train_epochs=3,  # 训练的轮数
    weight_decay=0.01,  # 权重衰减，用于防止过拟合
    logging_dir='./logs',  # 日志保存目录
    logging_steps=10,  # 每 10 个步骤记录一次日志
)

# 使用 Trainer API 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# 开始训练
trainer.train()

# 保存训练后的模型
trainer.save_model("./gpt2-finetuned")

# 测试模型并输出生成文本
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
model.eval()
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=1.0)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated text:\n", generated_text)