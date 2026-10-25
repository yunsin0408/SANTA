#!/usr/bin/env python3
"""Anonymized-query version of the same-domain split (see build_samedomain_split.py).

Reuses the EXISTING half A / half B document membership of data/pretrain_samedomain{SUFFIX}/ (read from its
sda_pairs.train.jsonl source_file values), so the anonymized arms use exactly the same documents per half
as the original-query arms -- only the queries/descriptions differ. SDA pairs come from data/pretrain_anonq/
(anonymized descriptions); MEP pairs come from data/pretrain/ unchanged, since MEP masks entities in the
document and never uses the query. Retrieval files come from data/retrieval_anonq/.

Writes data/pretrain_samedomain{SUFFIX}_anonq/ and data/retrieval_samedomain{SUFFIX}_anonq/ (new folders only).

Usage: python datasets/build_samedomain_split_anonq.py --suffix ""      # original 150/147 split
       python datasets/build_samedomain_split_anonq.py --suffix _alt    # alternate 143/154 split
"""
import argparse
import json
from pathlib import Path

import build_samedomain_split as base

REPO = Path(__file__).resolve().parents[3]
FOLDER = base.FOLDER


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="", help='"" for the original split, "_alt" for the alternate split')
    args = ap.parse_args()
    ref = REPO / "data" / "xml" / f"pretrain_samedomain{args.suffix}"
    half_a = {json.loads(l)["source_file"] for l in open(ref / "halfA" / FOLDER / "sda_pairs.train.jsonl")}
    half_b = {json.loads(l)["source_file"] for l in open(ref / "halfB" / FOLDER / "sda_pairs.train.jsonl")}

    def half_of(source_file):
        if source_file in half_a:
            return "halfA"
        assert source_file in half_b, source_file
        return "halfB"

    base.half_of = half_of  # split_* look up the module-level half_of

    # pretrain source dir: anonymized SDA pairs + original MEP pairs
    src = REPO / "data" / "xml" / "pretrain_samedomain_anonq_src" / FOLDER
    src.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        (src / f"sda_pairs.{split}.jsonl").write_text((REPO / "data" / "xml" / "pretrain_anonq" / FOLDER / f"sda_pairs.{split}.jsonl").read_text())
        (src / f"mep_pairs.{split}.jsonl").write_text((REPO / "data" / "xml" / "pretrain" / FOLDER / f"mep_pairs.{split}.jsonl").read_text())

    base.split_pretrain(src, REPO / "data" / "xml" / f"pretrain_samedomain{args.suffix}_anonq")
    base.split_retrieval(REPO / "data" / "xml" / "retrieval_anonq" / FOLDER, REPO / "data" / "xml" / f"retrieval_samedomain{args.suffix}_anonq")

    # the anonymized halves must hold exactly the reference halves' documents
    for half, ref_set in (("halfA", half_a), ("halfB", half_b)):
        got = {json.loads(l)["source_file"] for l in
               open(REPO / "data" / "xml" / f"pretrain_samedomain{args.suffix}_anonq" / half / FOLDER / "sda_pairs.train.jsonl")}
        assert got == ref_set, half


if __name__ == "__main__":
    main()
