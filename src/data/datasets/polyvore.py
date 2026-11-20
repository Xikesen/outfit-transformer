# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
import os
import cv2
import json
import random
import pickle
from tqdm import tqdm
from ..datatypes import (
    FashionItem, 
    FashionCompatibilityQuery, 
    FashionComplementaryQuery, 
    FashionCompatibilityData, 
    FashionFillInTheBlankData, 
    FashionTripletData
)
import numpy as np

from datasets import load_dataset


POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR = (
    "{dataset_dir}/precomputed_clip_embeddings"
)

metadata = load_dataset('owj0421/polyvore')['data']
item_id2metadata_idx = {
    j['item_id']: i for i, j in tqdm(enumerate(metadata))
}
metadata_idx2item_id = {
    j: i for i, j in tqdm(enumerate(item_id2metadata_idx))
}
# Metadata without images, used when embeddings are precomputed.
metadata_noimg = metadata.remove_columns(['image'])

_SPLIT_ALIASES = {'valid': 'validation', 'val': 'validation'}

def _hf_split(split):
    return _SPLIT_ALIASES.get(split, split)


def load_metadata(*args, **kwargs):
    # Metadata is loaded at import time.
    return metadata

def load_embedding_dict(dataset_dir):
    e_dir = POLYVORE_PRECOMPUTED_CLIP_EMBEDDING_DIR.format(dataset_dir=dataset_dir)
    filenames = [filename for filename in os.listdir(e_dir) if filename.endswith(".pkl")]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    
    all_ids, all_embeddings = [], []
    for filename in filenames:
        filepath = os.path.join(e_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            all_ids += data['ids']
            all_embeddings.append(data['embeddings'])
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Loaded {len(all_embeddings)} embeddings")
    
    all_embeddings_dict = {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}
    print(f"Created embeddings dictionary")
    
    return all_embeddings_dict


def load_item(item_id, embedding_dict: dict = None, load_image: bool = None) -> FashionItem:
    if load_image is None:
        load_image = embedding_dict is None
    m = (metadata if load_image else metadata_noimg)[item_id2metadata_idx[item_id]]

    image = m.get('image')
    if image is not None and image.mode != 'RGB':
        image = image.convert('RGB')

    return FashionItem(
        item_id=m['item_id'], image=image, category=m['category'], description=m['url_name'],
        embedding=embedding_dict[item_id] if embedding_dict else None
    )


class PolyvoreCompatibilityDataset(Dataset):

    def __init__(
        self,
        dataset_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        dataset_split: Literal['train', 'valid', 'test'] = 'train',
        embedding_dict: dict = None,
        **kwargs,
    ):
        dataset_split = _hf_split(dataset_split)
        # Modified >>>
        task = 'compatibility'
        self.data = load_dataset(
            'owj0421/polyvore-outfits', f'{dataset_type}_{task}', split=dataset_split
        )
        self.id_converter = {}
        task = 'default'
        set_infos = load_dataset(
            'owj0421/polyvore-outfits', f'{dataset_type}_{task}', split=dataset_split
        )
        for i in set_infos:
            for j in i['items']:
                self.id_converter[f"{i['set_id']}_{j['index']}"] = j['item_id']
        # <<<
        self.embedding_dict = embedding_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionCompatibilityData:
        label = int(self.data[idx]['label'])
        outfit = [
            load_item(self.id_converter[item_id], self.embedding_dict)
            for item_id in self.data[idx]['items']
        ]
        
        return FashionCompatibilityData(
            label=label,
            query=FashionCompatibilityQuery(outfit=outfit)
        )
        
class PolyvoreFillInTheBlankDataset(Dataset):

    def __init__(
        self,
        dataset_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        dataset_split: Literal['train', 'valid', 'test'] = 'train',
        embedding_dict: dict = None,
        **kwargs,
    ):
        dataset_split = _hf_split(dataset_split)
        # Modified >>>
        task = 'fill_in_the_blank'
        self.data = load_dataset(
            'owj0421/polyvore-outfits', f'{dataset_type}_{task}', split=dataset_split
        )
        self.id_converter = {}
        task = 'default'
        set_infos = load_dataset(
            'owj0421/polyvore-outfits', f'{dataset_type}_{task}', split=dataset_split
        )
        for i in set_infos:
            for j in i['items']:
                self.id_converter[f"{i['set_id']}_{j['index']}"] = j['item_id']
        # <<<
        self.embedding_dict = embedding_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionFillInTheBlankData:
        label = self.data[idx]['label']
        candidates = [
            load_item(self.id_converter[item_id], self.embedding_dict)
            for item_id in self.data[idx]['candidates']
        ]
        outfit = [
            load_item(self.id_converter[item_id], self.embedding_dict)
            for item_id in self.data[idx]['items']
        ]
        
        return FashionFillInTheBlankData(
            query=FashionComplementaryQuery(outfit=outfit, category=candidates[label].category),
            label=label,
            candidates=candidates
        )
    
        
class PolyvoreTripletDataset(Dataset):

    def __init__(
        self,
        dataset_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        dataset_split: Literal['train', 'valid', 'test'] = 'train',
        embedding_dict: dict = None,
        **kwargs,
    ):
        dataset_split = _hf_split(dataset_split)
        # Modified >>>
        task = 'default'
        self.data = load_dataset(
            'owj0421/polyvore-outfits', f'{dataset_type}_{task}', split=dataset_split
        )
        # <<<
        self.embedding_dict = embedding_dict
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> FashionTripletData:
        items = [
            load_item(item['item_id'], self.embedding_dict)
            for item in self.data[idx]['items']
        ]
        answer = items[random.randint(0, len(items) - 1)]
        outfit = [item for item in items if item != answer]
        
        return FashionTripletData(
            query=FashionComplementaryQuery(outfit=outfit, category=answer.category),
            answer=answer
        )
        

class PolyvoreItemDataset(Dataset):

    def __init__(
        self,
        embedding_dict: dict = None,
        load_image: bool = None,
        **kwargs,
    ):
        self.embedding_dict = embedding_dict
        self.load_image = load_image

    def __len__(self):
        return len(item_id2metadata_idx)
    
    def __getitem__(self, idx) -> FashionItem:
        item_id = metadata_idx2item_id[idx]
        item = load_item(item_id, embedding_dict=self.embedding_dict, load_image=self.load_image)

        return item
    
    def get_item_by_id(self, item_id):
        return load_item(item_id, embedding_dict=self.embedding_dict, load_image=self.load_image)
        
        
if __name__ == '__main__':
    # Test the dataset
    dataset = PolyvoreCompatibilityDataset(
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreFillInTheBlankDataset(
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreTripletDataset(
        dataset_type='nondisjoint',
        dataset_split='train'
    )
    print(len(dataset))
    print(dataset[0])
    
    dataset = PolyvoreItemDataset()
    print(len(dataset))
    print(dataset[0])