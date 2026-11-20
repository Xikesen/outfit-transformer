"""Baseline B1: zero-shot FashionCLIP retrieval.

For each held-out (test) outfit that has a usable title:
  query q = outfit title
  slots   = the coarse categories of the ground-truth outfit (oracle slot set,
            shared by all systems)
  for each slot, rank all catalog items of that category by cos(text(q), image(item))
  outfit m (m = 1..M) = the rank-m item of every slot

Metrics (Table 3 of the midway summary):
  Recall@K   fraction of GT items found in the top-K pool of their slot
  Slot hit   fraction of GT items that are the top-1 item of their slot
  CLIP score mean cos(text(q), image(item)) over returned items (outfit 1..M)
  Judge      mean CP score of the M returned outfits under --judge_checkpoint
  Valid      % outfits with (tops+bottoms or all-body) + shoes and no duplicate slot
  Diversity  mean pairwise (1 - cos) between the M outfits' items, per slot
  s/query    wall-clock per query (text encode + retrieval + judge scoring)

Usage:
  python -m src.run.4_baseline_b1 --titles_json /content/polyvore_outfit_titles.json \
      --judge_checkpoint checkpoints/<run>/epoch_N.pth --n_queries 2000
"""
import json
import os
import pathlib
import random
import re
import time
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ..data.datasets import polyvore
from ..data.datatypes import FashionCompatibilityQuery, FashionItem
from ..models.load import load_model

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
RESULT_DIR = SRC_DIR / 'results'
GENERIC_TITLE = re.compile(r'^(untitled|set|outfit|look|polyvore|style|fashion|my style|no title)?\s*#?\s*\d*$', re.I)


def parse_args():
    p = ArgumentParser()
    p.add_argument('--polyvore_dir', type=str, default='./datasets/polyvore')
    p.add_argument('--polyvore_type', type=str, default='nondisjoint')
    p.add_argument('--titles_json', type=str, required=True,
                   help='polyvore_outfit_titles.json (mvasil/polyvore-outfits)')
    p.add_argument('--judge_checkpoint', type=str, default=None,
                   help='CP model used for grading only')
    p.add_argument('--clip_name', type=str, default='patrickjohncyh/fashion-clip')
    p.add_argument('--n_queries', type=int, default=2000)
    p.add_argument('--M', type=int, default=3)
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_name', type=str, default='b1_zero_shot')
    return p.parse_args()


def get_title(entry):
    if isinstance(entry, dict):
        return (entry.get('title') or entry.get('url_name') or '').strip()
    return str(entry or '').strip()


def usable(title):
    t = title.strip()
    return len(t) >= 3 and not GENERIC_TITLE.match(t) and re.search(r'[a-zA-Z]{3,}', t) is not None


def is_valid(slots):
    s = set(slots)
    if len(s) != len(slots):
        return False
    core = ({'tops', 'bottoms'} <= s) or ('all-body' in s)
    return core and 'shoes' in s


@torch.no_grad()
def main(args):
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test outfits and titles
    with open(args.titles_json) as f:
        titles = json.load(f)
    test = load_dataset('owj0421/polyvore-outfits', f'{args.polyvore_type}_default', split='test')
    outfits = []
    for row in test:
        t = get_title(titles.get(row['set_id'], ''))
        if usable(t):
            outfits.append((row['set_id'], t, [it['item_id'] for it in row['items']]))
    print(f'{len(outfits)}/{len(test)} test outfits have a usable title')

    # Catalog: all test items, grouped by category
    emb = polyvore.load_embedding_dict(args.polyvore_dir)
    cat_items = defaultdict(list)
    seen = set()
    for row in test:
        for it in row['items']:
            iid = it['item_id']
            if iid in seen or iid not in emb:
                continue
            seen.add(iid)
            cat = polyvore.metadata_noimg[polyvore.item_id2metadata_idx[iid]]['category']
            cat_items[cat].append(iid)
    item_cat = {iid: c for c, ids in cat_items.items() for iid in ids}
    cat_img = {}
    for c, ids in cat_items.items():
        m = torch.tensor(np.stack([emb[i][:512] for i in ids]), dtype=torch.float32, device=device)
        cat_img[c] = (ids, F.normalize(m, dim=-1))
    print('catalog:', {c: len(v[0]) for c, v in cat_img.items()})

    random.shuffle(outfits)
    outfits = outfits[:args.n_queries]

    tok = CLIPTokenizer.from_pretrained(args.clip_name)
    txt = CLIPTextModelWithProjection.from_pretrained(args.clip_name).to(device).eval()
    judge = None
    if args.judge_checkpoint:
        judge = load_model(model_type='clip', checkpoint=args.judge_checkpoint).eval()

    agg = defaultdict(list)
    examples = []
    for set_id, title, gt_ids in tqdm(outfits, desc='B1'):
        t0 = time.time()
        inp = tok([title], padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
        q = F.normalize(txt(**inp).text_embeds, dim=-1)[0]  # (512,)

        gt_by_slot = defaultdict(list)
        for g in gt_ids:
            if g in item_cat:
                gt_by_slot[item_cat[g]].append(g)
        slots = list(gt_by_slot.keys())
        if not slots:
            continue

        pools, pool_sims = {}, {}
        for s in slots:
            ids, mat = cat_img[s]
            sims = mat @ q
            k = min(max(args.K, args.M), len(ids))
            top = torch.topk(sims, k)
            pools[s] = [ids[j] for j in top.indices.tolist()]
            pool_sims[s] = top.values.tolist()

        M = min(args.M, min(len(pools[s]) for s in slots))
        returned = [[pools[s][m] for s in slots] for m in range(M)]

        judge_scores = []
        if judge is not None:
            qs = [FashionCompatibilityQuery(outfit=[FashionItem(item_id=i, embedding=emb[i]) for i in o]) for o in returned]
            judge_scores = judge(qs, use_precomputed_embedding=True).view(-1).tolist()
        latency = time.time() - t0


        n_gt = sum(len(v) for v in gt_by_slot.values())
        agg[f'recall@{args.K}'].append(sum(g in pools[s][:args.K] for s, gs in gt_by_slot.items() for g in gs) / n_gt)
        agg['slot_hit'].append(sum(g == pools[s][0] for s, gs in gt_by_slot.items() for g in gs) / n_gt)
        agg['clip_score'].append(float(np.mean([pool_sims[s][m] for s in slots for m in range(M)])))
        if judge_scores:
            agg['judge_compat'].append(float(np.mean(judge_scores)))
        agg['valid_pct'].append(100.0 * is_valid(slots))
        if M > 1:
            d = []
            for s in slots:
                v = F.normalize(torch.tensor(np.stack([emb[pools[s][m]][:512] for m in range(M)])), dim=-1)
                sim = v @ v.T
                iu = torch.triu_indices(M, M, 1)
                d.append((1 - sim[iu[0], iu[1]]).mean().item())
            agg['diversity'].append(float(np.mean(d)))
        agg['sec_per_query'].append(latency)
        if len(examples) < 20:
            examples.append({'set_id': set_id, 'title': title, 'slots': slots, 'gt': gt_ids, 'returned': returned,
                             'judge': judge_scores})

    result = {k: float(np.mean(v)) for k, v in agg.items()}
    result['n_queries'] = len(agg['sec_per_query'])
    result['judge_checkpoint'] = args.judge_checkpoint
    print(json.dumps(result, indent=2))
    out = RESULT_DIR / args.out_name
    os.makedirs(out, exist_ok=True)
    with open(out / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    with open(out / 'examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    print('saved to', out)


if __name__ == '__main__':
    main(parse_args())
