import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

plt.style.use('seaborn-whitegrid')


class Dataset:
    def __init__(self, data_dir=None):
        self.train_data = self.load_data(data_dir, data_type="train")
        self.valid_data = self.load_data(data_dir, data_type="valid")
        self.test_data = self.load_data(data_dir, data_type="test")

    @staticmethod
    def load_data(data_dir, data_type):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    @staticmethod
    def descriptive_statistics(data, info):
        directed_nodes = dict()
        in_directed_nodes = dict()
        edges = dict()

        # 1. Iterate over triples and create
        # Given (e_i,r_j,e_k)
        # 1.2. e_i => r_j, e_k mapping => Directed Mapping
        # 1.3. e_k => r_j, e_i mapping => Reverse directed mapping
        # 1.4. r_j => e_k, e_i => Relation mapping

        for i in data:
            h, r, t = i[0], i[1], i[2]
            directed_nodes.setdefault(h, []).append((r, t))
            in_directed_nodes.setdefault(t, []).append((h, r))
            edges.setdefault(r, []).append((h, t))

        # Unique entities
        unique_entities = set(directed_nodes.keys()).union(set(in_directed_nodes.keys()))

        print(f'############### DESCRIPTION {info} ###############')
        print(f'Number of triples = {len(data)}')
        print(f'Number of unique entities = {len(unique_entities)}')
        # print(f'Number of unique head entities = {len(unique_head_entities)}')
        # print(f'Number of unique tail entities = {len(unique_tail_entities)}')
        print(f'Number of relations = {len(edges)}')

        print('\n')
        # Degree of nodes.
        out_degrees = np.array([len(v) for k, v in directed_nodes.items()])
        in_degrees = np.array([len(v) for k, v in in_directed_nodes.items()])
        print(f'{in_degrees.mean():.3f}+-{in_degrees.std():.3f} indegree of a node.')
        print(f'{out_degrees.mean():.3f}+-{out_degrees.std():.3f} outdegree of a node.')
        print('#' * 10)

        """
        for k, v in edges.items():
            G.add_edges_from(list(v), label=k)

        degree_sequence = sorted([degree for node, degree in G.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        plt.bar(deg, cnt)
        plt.title(f"Degree Histogram {info}")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        plt.tight_layout()
        plt.show()
        """

        """
        
        # = Unable to allocate 12.3 GiB for an array with shape (40559, 40559) and data type float64
        L = nx.normalized_laplacian_matrix(G)
        e = np.linalg.eigvals(L.A)
        print("Largest eigenvalue:", max(e))
        print("Smallest eigenvalue:", min(e))
        plt.hist(e)  # histogram with 100 bins
        #plt.xlim(0, 2)  # eigenvalues between 0 and 2
        plt.show()
        """
        """
        Seem to take time
        print("Betweenness")
        b = nx.betweenness_centrality(G)
        for v in G.nodes():
            print(f"{v:2} {b[v]:.3f}")

        print("Degree centrality")
        d = nx.degree_centrality(G)
        for v in G.nodes():
            print(f"{v:2} {d[v]:.3f}")

        print("Closeness centrality")
        c = nx.closeness_centrality(G)
        for v in G.nodes():
            print(f"{v:2} {c[v]:.3f}")
        """

    @staticmethod
    def describe_oov(data, entities,info):
        triples_contain_oov_entity_both_position = []
        triples_contain_oov_entity_head = []
        triples_contain_oov_entity_tail = []

        triples_contain_oov = []
        for i in data:
            h, r, t = i[0], i[1], i[2]

            # 1. The head or tail entity is an OOV entity.
            if (h not in entities) or (t not in entities):
                triples_contain_oov.append(i)

                # 2. The head entity is an OOV entity.
                if h not in entities:
                    triples_contain_oov_entity_head.append(i)

                # 3. The tail entity is an OOV entity.
                if t not in entities:
                    triples_contain_oov_entity_tail.append(i)

                # 4. The head and the tail entities are OOV entities.
                if (h not in entities) and (t not in entities):
                    triples_contain_oov_entity_both_position.append(i)
        print('\n')
        print(f'############### DESCRIPTION {info} ###############')
        print(
            f'{len(triples_contain_oov)} triples contain OOV entities. \t {len(triples_contain_oov) / len(data) * 100:.3f} % of triples contain OOV entities')

        print(f'{len(triples_contain_oov_entity_head)} triples contain OOV entities in the head position.')
        print(f'{len(triples_contain_oov_entity_tail)} triples contain OOV entities in the tail position.')
        print(f'{len(triples_contain_oov_entity_tail)} triples contain OOV entities in the head and tail positions.')

        print('#' * 100)
        print(
            f'Freq. of relations occurring with oov entities: \n{collections.Counter([i[1] for i in triples_contain_oov])}')

        print(
            f'Freq. of relations occurring with oov entities in the head position: \n{collections.Counter([i[1] for i in triples_contain_oov_entity_head])}')

        print(
            f'Freq. of relations occurring with oov entities in the tail position: \n{collections.Counter([i[1] for i in triples_contain_oov_entity_tail])}')

        print(
            f'Frequency of relations occurring with oov entities in the head and tail position: \n{collections.Counter([i[1] for i in triples_contain_oov_entity_tail])}')
