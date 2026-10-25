# 02 Pooling fix: mean pooling + cosine

Launcher: [`run.sh`](run.sh). Results: [`results/02_pooling_fix/`](../../results/02_pooling_fix/). Config A, used throughout, is
`--pooling mean --normalize True --temperature 0.05` (mean pooling, cosine, tau 0.05).

**Results may vary by run.** GPU training is not bit-exact, so reruns with the same seed can move a cell's test MRR by a few
hundredths (up to ~0.05 on small test sets). 

### Pooling / normalization fix (S1000D_spec_sample target, CodeT5, seed 42)
Test MRR, chance 0.1736:

| Pretrain source -> S1000D_spec_sample | Baseline: first-step state, dot (exp 01 `cnt_p1024_s42`) | Mean pooling, dot (`fixB_meandot_p1024_s42`) | Mean pooling, cosine (`fixA_meancos_p1024_s42`) | Mean pooling, cosine, `p_max_len` 256 (`fixC_meancos_p256_s42`) |
| --- | --- | --- | --- | --- |
| BIKE_7 | 0.1538 | 0.8560 | 0.8984 | 0.5385 |
| FOSSIG | 0.1976 | 0.8476 | 0.9286 | 0.5923 |
| s1kd-tools-doc | 0.1756 | 0.9206 | 0.9524 | 0.5481 |

- Changing the pooling alone moves the score from chance to 0.85–0.95, so the chance-level results of exp 01 come from the embedding, not from the data volume or the tokenizer.
- Config A over 3 seeds (`fixA_meancos_p1024_s{42,1,2}`, 3 sources each): mean 0.917, sd 0.059.
- **No-pretraining** (`nopre_meancos_p1024_s{42,1,2,3,4}`, base CodeT5 finetuned on S1000D with config A, launcher `experiments/common/run_nopretrain.sh`): 0.917 / 0.976 / 0.976 / 0.881 / 0.952
  (mean 0.940, sd 0.041). This is at least as high as with SDA/MEP pretraining (0.917); the 0.02 gap is inside the noise (about 0.9 standard errors), this benchmark shows no benefit from pretraining. It may be because the task is too easy (BM25 = 1.0).

### Fix on the other three targets, and `p_max_len` 2048 (CodeT5, mean pooling + cosine tau 0.05, seed 42)

Extends the pooling fix beyond S1000D_spec_sample to the 9 pairs targeting BIKE_7, FOSSIG, and s1kd-tools-doc, and checks whether a
longer passage window (2048 vs. 1024) helps on these longer-document targets. Folders `02_pooling_fix/allA_meancos_p1024_s42/`,
`02_pooling_fix/allA_meancos_p2048_s42/`, no-pretrain control `02_pooling_fix/nopreAll_meancos_p1024_s{42,1,2}/`.

| Pair | Chance | p1024 | p2048 |
| --- | --- | --- | --- |
| BIKE_7 -> FOSSIG | 0.7500 | 0.7500 | 0.7500 |
| S1000D_spec_sample -> FOSSIG | 0.7500 | 1.0000 | 1.0000 |
| s1kd-tools-doc -> FOSSIG | 0.7500 | 1.0000 | 0.7500 |
| BIKE_7 -> s1kd-tools-doc | 0.6111 | 0.8333 | 1.0000 |
| FOSSIG -> s1kd-tools-doc | 0.6111 | 0.8333 | 1.0000 |
| S1000D_spec_sample -> s1kd-tools-doc | 0.6111 | 1.0000 | 1.0000 |
| FOSSIG -> BIKE_7 | 0.4083 | 0.5194 | 0.9167 |
| S1000D_spec_sample -> BIKE_7 | 0.4083 | 0.9167 | 0.8333 |
| s1kd-tools-doc -> BIKE_7 | 0.4083 | 0.6583 | 0.8056 |

Mean per target (p1024 / p2048): BIKE_7 0.698 / 0.852, s1kd-tools-doc 0.889 / 1.000, FOSSIG 0.917 / 0.833. No-pretraining control (p1024, 3 seeds):
BIKE_7 0.566, FOSSIG 0.917, s1kd-tools-doc 0.778.

- **The pooling fix generalizes**: 8 of 9 pairs at p1024 and 7 of 9 at p2048 score above chance (BIKE_7 -> FOSSIG stays at chance at both
  lengths, s1kd-tools-doc -> FOSSIG at p2048), and every target's mean is above chance.
- **Caveat**: test sets are 6 / 3 / 2 documents, so individual cells move by 0.05-0.25 per query and should be read as directional, not precise.
- **2048 looks better than 1024 on average** (BIKE_7 and s1kd-tools-doc both improve; FOSSIG, only 2 test docs, moves the other way on
  one query). Consistent with fixC vs. fixA above (p256 well below p1024 on S1000D_spec_sample).

### Why the embeddings collapse, and why mean pooling fixes it

**Background:** T5 / CodeT5 is a "read one text, write another" model (used for translation or summarization). The encoder reads the
input and produces one vector per token. The decoder then writes the output one token at a time. Before each token it computes an internal state (768 numbers, looking
back at the encoder's vectors), and a vocabulary head turns that state into probabilities for the next token:

```
input text ----> [encoder] ----> one vector per input token (the text, "understood")
                                         |  the decoder looks back at these
                                         v
start token ---> [decoder] ---> internal state (768 numbers) ---> vocab head ---> predicts output token 1
token 1 -------> [decoder] ---> internal state                ---> vocab head ---> predicts output token 2
...
```

**How the baseline builds a vector (`pooling=first`, SANTA's original setting, `model.py`).** Steps:

1. The encoder reads the whole document.
2. The decoder is given one fixed "start" token, identical for every document.
3. The decoder's output at that single step is taken as the document's vector.

In code (`encode_q` / `encode_p` in `model.py`, same for queries and documents):

```python
decoder_input_ids = torch.zeros((batch, 1))        # one fixed start token (id 0) for every text
out = model(**items, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
hidden = out.decoder_hidden_states[-1]             # decoder's last-layer internal state
reps = hidden[:, 0, :]                             # position 0 = that single step -> the embedding
```

With `pooling=mean` this whole decoder step is skipped: the embedding is the average of the encoder's per-token vectors instead.

**Evidence** (diagnostics in exp 01's `results/01_baselines/t5_vs_codet5/README.md`, branch `exp/01-baselines`):
- Even an **untrained** CodeT5 does this: any two documents' vectors have cosine 1.0000, and the difference between documents is only 0.6% of the vector's size.
  Queries collapse too (cosine 0.9999).
- The shared part is concentrated in a few dimensions: 45% of the mean vector's energy sits in a single dimension (dim 567). This big common offset adds about the same
  amount to every document's score and hides the real differences.
- Subtracting the shared part lifts the untrained CodeT5's MRR from 0.17 to 0.60.

**The logic of mean pooling.** Skip the decoder step. Average the encoder's output over every token of the document and use that as the vector. Each
token's output already carries the meaning of that token, so documents with different content get different averages.

**What cosine adds.** It rescales every vector to length 1, so only direction matters. The score no longer depends on vector length, and the model cannot change scores
by changing lengths; it has to separate documents by direction.
