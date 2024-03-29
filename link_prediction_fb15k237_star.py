from util.data import Dataset
from util.eval import Evaluator
import json
import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import numpy as np

# Link prediction performances of RESCAL, ComplEx, ConvE, DistMult and TransE on FB15k-237^* (out-of-vocabulary entities are removed)
models = ['rescal', 'complex', 'conve', 'distmult', 'transe']

for m in models:
    if m == 'conex':
        """ """
        raise NotImplementedError()
    else:
        # 1. Load pretrained model via LibKGE
        checkpoint = load_checkpoint(f'pretrained_models/FB15K-237/fb15k-237-{m}.pt')
        model = KgeModel.create_from(checkpoint)

        # 3. Create mappings.
        # 3.1 Entity index mapping.
        entity_idxs = {e: e_idx for e, e_idx in zip(model.dataset.entity_ids(), range(len(model.dataset.entity_ids())))}
        # 3.2 Relation index mapping.
        relation_idxs = {r: r_idx for r, r_idx in
                         zip(model.dataset.relation_ids(), range(len(model.dataset.relation_ids())))}
    # 2. Load Dataset
    dataset = Dataset(data_dir=f'KGs/FB15K-237*/')

    # 4. Subject-Predicate to Object mapping and Predicate-Object to Subject mapping. This will be used at computing filtering ranks.
    sp_vocab, so_vocab, po_vocab = dataset.get_mappings(dataset.train_data + dataset.valid_data + dataset.test_data,
                                                        entity_idxs=entity_idxs, relation_idxs=relation_idxs)

    ev = Evaluator(entity_idxs=entity_idxs, relation_idxs=relation_idxs, sp_vocab=sp_vocab, so_vocab=so_vocab,
                   po_vocab=po_vocab)

    lp_results = ev.filtered_link_prediction(dataset.test_data, model)
    with open(f'fb15k-237_*_{m}_results.json', 'w') as file_descriptor:
        json.dump(lp_results, file_descriptor, indent=2)
