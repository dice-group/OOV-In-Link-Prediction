import os
import json

import numpy as np
from scipy.stats import wilcoxon


def get_results(path='results', dataset='wnrr'):
    r = dict()
    for i in (i for i in os.listdir(path)):
        if dataset in i:
            with open(path + '/' + i, 'r') as json_file:
                settings = json.load(json_file)
                r[i] = settings
    return r


for dataset in ['wnrr', 'fb15k-237']:

    results = get_results(dataset=dataset)

    diff = []
    for model in ['rescal', 'conve', 'distmult', 'complex']:
        p1 = f'{dataset}_*_{model}_results.json'
        p2 = f'{dataset}_{model}_results.json'
        corrected, not_corrected = results[p1], results[p2]

        # print(f'Hypothesis Testing for {model} on {dataset} and {dataset}*')
        for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
            diff.append((corrected[metric] - not_corrected[metric]))

    diff = np.array(diff)
    w, p_value = wilcoxon(diff, mode='exact')
    # print(f'Differences across models : {diff.mean() * 100} +- {diff.std() * 100} % on {dataset} and {dataset}*')
    print(f'\nReject the null hypothesis at a confidence level of {p_value} on {dataset} and {dataset}*')
