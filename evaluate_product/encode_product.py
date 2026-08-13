"""
Encodes a query or product-doc jsonl file (as produced by build_test_product.py)
into a pickled (embeddings, ids) tuple, in the format evaluate_product.py expects
via --pickle_query_data / --pickle_doc_data.

This script didn't exist in the original repo (shell/index-product.sh was an empty
file) -- it mirrors the encode/pooling logic already used for dev-checkpoint
selection in product_best_dev/evaluate_shop_pretrain.py (T5ForConditionalGeneration,
zero decoder_input_ids, last hidden state at position 0 as the pooled vector).
"""
import argparse
import json
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_data(file_path, text_field):
    data = []
    with open(file_path) as f:
        for line in f:
            js = json.loads(line.strip())
            data.append({"text": js[text_field], "id": js["id"]})
    return data


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [ex["text"] for ex in batch]
        ids = [ex["id"] for ex in batch]
        outputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return outputs["input_ids"], outputs["attention_mask"].bool(), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="query or doc jsonl from build_test_product.py")
    parser.add_argument("--data_type", type=str, choices=["query", "doc"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="path to write the pickled (vecs, ids) tuple")
    parser.add_argument("--max_length", type=int, default=None, help="defaults to 50 for query, 240 for doc")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    text_field = "query" if args.data_type == "query" else "product"
    max_length = args.max_length or (50 if args.data_type == "query" else 240)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.eval()

    dataset = TextDataset(load_data(args.input, text_field))
    collator = Collator(tokenizer, max_length)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    all_vecs = []
    all_ids = []
    with torch.no_grad():
        for input_ids, attention_mask, ids in tqdm(loader, desc=f"encoding {args.data_type}"):
            decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=device)
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            vec = out[2][-1].squeeze(1)
            all_vecs.append(vec.cpu().numpy())
            all_ids.extend(ids)

    all_vecs = np.concatenate(all_vecs, 0)
    with open(args.output, "wb") as f:
        pickle.dump((all_vecs, all_ids), f)
    print(f"wrote {args.output}: {all_vecs.shape[0]} vectors")


if __name__ == "__main__":
    main()
