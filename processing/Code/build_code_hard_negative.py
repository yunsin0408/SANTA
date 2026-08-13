import json
import random
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
from transformers import RobertaTokenizer

# Merges a mined-negatives trec file (produced by running openmatch retrieval with
# the *pretrained, not-yet-finetuned* SANTA checkpoint over the finetuning train
# split -- see shell/mine-hard-negatives-code.sh) back onto the raw train.jsonl,
# to build a finetuning file with populated `negatives`, matching the tokenization
# convention of build_code.py (which always writes `negatives: []`).
#
# Paper (Appendix A.4): "We use inbatch negatives with one hard negative for
# finetuning and the hard negative is randomly sampled from the top 100 retrieved
# negative codes by pretrained SANTA." Depth/corpus/sampling-seed specifics beyond
# that quote are this script's own judgment calls -- see shell/PIPELINE.md.


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_doc_pool(path):
    pool = {}
    with open(path) as f:
        for line in f:
            js = json.loads(line.strip())
            pool[js['id']] = js['code']
    return pool


def load_trec(path):
    # TREC format: <query_id> Q0 <doc_id> <rank> <score> <run_id>, already sorted
    # by descending score (see openmatch.utils.save_as_trec).
    ranked = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, did = parts[0], parts[1], parts[2]
            ranked[qid].append(did)
    return ranked


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                         help='raw train.jsonl (url, code_tokens/function_tokens, docstring_tokens)')
    parser.add_argument('--trec', type=str, required=True,
                         help='trec file: train queries retrieved against the mining corpus by the pretrained checkpoint')
    parser.add_argument('--doc_pool', type=str, required=True,
                         help='doc-side jsonl of the mining corpus (id -> code), built via build_test_code.py --data_type doc')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_hard_negatives', type=int, default=1,
                         help='paper: "one hard negative" per example')
    parser.add_argument('--top_k', type=int, default=100,
                         help='paper: "top 100 retrieved negative codes"')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

    data = load_jsonl(args.input)
    doc_pool = load_doc_pool(args.doc_pool)
    ranked = load_trec(args.trec)

    n_no_candidates = 0
    with open(args.output, 'w') as f:
        for item in tqdm(data):
            url = item['url']
            if 'code_tokens' in item:
                code = ' '.join(item['code_tokens'])
            else:
                code = ' '.join(item['function_tokens'])
            nl = ' '.join(item['docstring_tokens'])

            # Drop the query's own positive if the retriever happened to surface it,
            # then keep the top_k true negatives to sample from.
            candidates = [did for did in ranked.get(url, []) if did != url][:args.top_k]
            k = min(args.num_hard_negatives, len(candidates))
            if k == 0:
                n_no_candidates += 1
                continue
            sampled_ids = random.sample(candidates, k)
            negative_texts = [doc_pool[did] for did in sampled_ids if did in doc_pool]
            if not negative_texts:
                n_no_candidates += 1
                continue

            query = tokenizer.encode(nl, add_special_tokens=False, max_length=50, truncation=True)
            positives = tokenizer([code], add_special_tokens=False, max_length=256,
                                   truncation=True, padding=False)['input_ids']
            negatives = tokenizer(negative_texts, add_special_tokens=False, max_length=256,
                                   truncation=True, padding=False)['input_ids']

            group = {
                'query': query,
                'positives': positives,
                'negatives': negatives,
            }
            f.write(json.dumps(group) + '\n')

    print(f"wrote {len(data) - n_no_candidates} examples, "
          f"skipped {n_no_candidates} with no mined negative candidates")


if __name__ == '__main__':
    main()
