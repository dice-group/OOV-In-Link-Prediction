from util.data import Dataset

kg = ['WN18RR', 'FB15k-237', 'YAGO3-10']
for i in kg:
    dataset = Dataset(data_dir=f'KGs/{i}/')
    # Get all entities from train set.
    entities = set(dataset.get_entities(dataset.train_data))
    dataset.describe_oov(dataset.test_data, entities, info=f'{i}-Test set')
    dataset.describe_oov(dataset.valid_data, entities, info=f'{i}-Val set')

# Cleaned datasets
kg = ['WN18RR*', 'FB15k-237*', 'YAGO3-10*']
for i in kg:
    dataset = Dataset(data_dir=f'KGs/{i}/')
    # Get all entities from train set.
    entities = set(dataset.get_entities(dataset.train_data))
    dataset.describe_oov(dataset.test_data, entities, info=f'{i}-Test set')
    dataset.describe_oov(dataset.valid_data, entities, info=f'{i}-Val set')
