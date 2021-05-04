from util.data import Dataset

kg = ['WN18RR', 'FB15K-237']#, 'YAGO3-10']
for i in kg:
    dataset = Dataset(data_dir=f'KGs/{i}/')
    dataset.descriptive_statistics(dataset.train_data, info=f'{i}-Train set')
    dataset.descriptive_statistics(dataset.valid_data, info=f'{i}-Valid set')
    dataset.descriptive_statistics(dataset.test_data, info=f'{i}-Test set')
