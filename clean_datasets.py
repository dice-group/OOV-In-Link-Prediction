from util.data import Dataset
from typing import Generator, List


def clean_dataset(triples: List, entities, relations) -> Generator:
    for t in triples:

        s, p, o = t

        if (s in entities) and (p in relations) and (o in entities):
            yield t


def store(triples, full_path):
    with open(full_path, 'w') as writer:
        for t in triples:
            s, p, o = t
            t_str = s + '\t' + p + '\t' + o + '\n'
            writer.write(t_str)


kg = ['WN18RR', 'FB15k-237', 'YAGO3-10']
for i in kg:
    data_dir = f'KGs/{i}/'
    clean_data_dir = f'KGs/{i}*/'
    dataset = Dataset(data_dir=data_dir)
    print(data_dir)

    clean_valid_set = clean_dataset(dataset.valid_data, entities=dataset.get_entities(dataset.train_data),
                                    relations=dataset.get_relations(dataset.train_data))
    clean_test_set = clean_dataset(dataset.test_data, entities=dataset.get_entities(dataset.train_data),
                                   relations=dataset.get_relations(dataset.train_data))

    store(dataset.train_data, clean_data_dir + 'train.txt')  # Train set
    store(clean_valid_set, clean_data_dir + 'valid.txt')  # Cleaned valid set
    store(clean_test_set, clean_data_dir + 'test.txt')  # Clean test set
