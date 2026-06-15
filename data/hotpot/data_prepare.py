from datasets import load_dataset
import json
from pathlib import Path

data_files = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"]
}

ds = load_dataset("parquet", data_files=data_files)

print(f"训练集: {len(ds['train'])} 条")
print(f"验证集: {len(ds['validation'])} 条")

# 保存 QA 数据到当前文件夹
for split in ['train', 'validation']:
    filename = 'train.jsonl' if split == 'train' else 'eval.jsonl'
    with open(filename, 'w', encoding='utf-8') as f:
        for item in ds[split]:
            f.write(json.dumps({
                "id": item['id'],
                "question": item['question'],
                "answer": item['answer'],
                "aliases": []
            }, ensure_ascii=False) + '\n')
    print(f"✓ {filename} 已保存")

# 构建语料库到当前文件夹
docs = set()
for split in ['train', 'validation']:
    for item in ds[split]:
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            docs.add((title, ' '.join(sentences)))

with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for i, (title, text) in enumerate(docs):
        f.write(json.dumps({
            "id": f"doc_{i}",
            "title": title,
            "text": text[:2000]
        }, ensure_ascii=False) + '\n')

print(f"✓ corpus.jsonl 已保存 ({len(docs)} 篇文档)")
