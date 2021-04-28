import torch
import numpy as np
import time


class Evaluator:
    def __init__(self, *, entity_idxs, relation_idxs, sp_vocab, so_vocab, po_vocab):
        self.entity_idxs = entity_idxs
        self.relation_idxs = relation_idxs
        self.sp_vocab = sp_vocab
        self.po_vocab = po_vocab
        self.so_vocab = so_vocab

    def filtered_link_prediction(self, triples, model):
        results = dict()

        num_param = sum([p.numel() for p in model.parameters()])
        results['Number_param'] = num_param
        reciprocal_ranks = []
        hits = dict()
        reciprocal_rank_per_relation = dict()

        for t in triples:
            # 1. Get a test triple
            s_str, p_str, o_str = t[0], t[1], t[2]
            # 2. Map (1) to indexes.
            s_idx = self.entity_idxs[s_str]
            p_idx = self.relation_idxs[p_str]
            o_idx = self.entity_idxs[o_str]

            # 3. Convert index into tensor
            s1 = torch.Tensor([s_idx]).long()
            p1 = torch.Tensor([p_idx]).long()

            # 4. Compute the filtered rank of the missing tail entity
            pred_tail = model.score_sp(s1, p1)  # scores of all objects for (s,p,?)
            pred_tail = pred_tail[0]
            # 4.1 {x | (s,p,x) in Train or Valid or Test }
            filt = self.sp_vocab[(s_idx, p_idx)]
            target_value = pred_tail[o_idx].item()
            pred_tail[filt] = 0.0
            pred_tail[o_idx] = target_value
            _, sort_tail_idxs = torch.sort(pred_tail, descending=True)
            sort_tail_idxs = sort_tail_idxs.cpu().numpy()
            rank_of_missing_tail_entity = np.where(sort_tail_idxs == o_idx)[0][0]

            # 5. Compute the filtered rank of missing head entity
            p1 = torch.Tensor([p_idx]).long()
            o1 = torch.Tensor([o_idx]).long()
            pred_head = model.score_po(p1, o1)
            pred_head = pred_head[0]

            # 5.1 {x | (x,p,o) in Train or Valid or Test }
            filt = self.po_vocab[(p_idx, o_idx)]
            target_value = pred_head[s_idx].item()
            pred_head[filt] = 0.0
            pred_head[s_idx] = target_value
            _, sort_head_idxs = torch.sort(pred_head, descending=True)
            sort_head_idxs = sort_head_idxs.cpu().numpy()
            rank_of_missing_head_entity = np.where(sort_head_idxs == s_idx)[0][0]

            # Add 1 because np.where(.) start from 0. Perfect prediction (1/0) hence ill defined.
            rank_of_missing_tail_entity += 1
            rank_of_missing_head_entity += 1

            reciprocal_rank_per_relation.setdefault(p_str, []).append(
                (1 / rank_of_missing_head_entity) + (1 / rank_of_missing_tail_entity))

            for hits_level in range(1, 11):
                I = 1 if rank_of_missing_head_entity <= hits_level else 0
                I += 1 if rank_of_missing_tail_entity <= hits_level else 0
                if I > 0:
                    hits.setdefault(hits_level, []).append(I)

            reciprocal_ranks.append(1 / rank_of_missing_tail_entity + 1 / rank_of_missing_head_entity)

        reciprocal_ranks = np.array(reciprocal_ranks).sum() / (2 * len(triples))
        results['MRR'] = reciprocal_ranks
        for hits_level, scores in hits.items():
            results[f'Hits@{hits_level}'] = sum(scores) / (2 * len(triples))

        # Link prediction per relation
        for k, v in reciprocal_rank_per_relation.items():
            # ranks => sum of head and tail entities given a relation
            results[f'MRR_{k}'] = sum(v) / (2 * len(v))
        return results
