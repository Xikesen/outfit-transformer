"""FITB accuracy with the compatibility model.

Each candidate is added to the partial outfit and scored; the highest
scoring candidate is the prediction.

Usage:
    python -m src.run.2_test_fitb_cp --checkpoint checkpoints/<run>/epoch_N.pth
"""
import json
import os
import pathlib
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import collate_fn
from ..data.datasets import polyvore
from ..data.datatypes import FashionCompatibilityQuery
from ..models.load import load_model
from ..utils.utils import seed_everything

SRC_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
RESULT_DIR = SRC_DIR / 'results'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['original', 'clip'], default='clip')
    parser.add_argument('--polyvore_dir', type=str, default='./datasets/polyvore')
    parser.add_argument('--polyvore_type', type=str, choices=['nondisjoint', 'disjoint'], default='nondisjoint')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_sz', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--demo', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def main(args):
    embedding_dict = polyvore.load_embedding_dict(args.polyvore_dir)
    data = polyvore.PolyvoreFillInTheBlankDataset(
        dataset_type=args.polyvore_type, dataset_split=args.split, embedding_dict=embedding_dict
    )
    loader = DataLoader(data, batch_size=args.batch_sz, shuffle=False,
                        num_workers=args.n_workers, collate_fn=collate_fn.fitb_collate_fn)
    model = load_model(model_type=args.model_type, checkpoint=args.checkpoint)
    model.eval()

    n_correct, n_total, t0 = 0, 0, time.time()
    pbar = tqdm(loader, desc='[Test] FITB via CP scores')
    for i, batch in enumerate(pbar):
        if args.demo and i > 2:
            break
        queries = []
        for q, cands in zip(batch['query'], batch['candidates']):
            for c in cands:
                queries.append(FashionCompatibilityQuery(outfit=list(q.outfit) + [c]))
        scores = model(queries, use_precomputed_embedding=True).view(len(batch['label']), -1)  # (B, 4)
        preds = scores.argmax(dim=-1).cpu()
        labels = torch.tensor(batch['label'])
        n_correct += (preds == labels).sum().item()
        n_total += len(labels)
        pbar.set_postfix(acc=n_correct / n_total)

    result = {'fitb_acc_cp': n_correct / n_total, 'n': n_total, 'seconds': time.time() - t0}
    print(f'[Test] FITB (CP scoring) --> {result}')
    out_dir = RESULT_DIR / pathlib.Path(args.checkpoint).parent.name
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f'fitb_cp_{args.split}.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)
