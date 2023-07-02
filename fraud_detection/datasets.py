import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class EllipticDataset:
    def __init__(self, config):
        self.features_df = pd.read_csv(config.features_path, header=None)
        self.edges_df = pd.read_csv(config.edges_path)
        self.labels_df = pd.read_csv(config.classes)
        self.labels_df["class"] = self.labels_df["class"].map({'unknown': 2, '1': 1, '2': 0})
        self.merged_df = self.merge()
        self.edge_index = self._edge_index()
        self.edge_weights = self._edge_weights()
        self.node_features = self._node_features()
        self.labels = self._labels()
        self.classified_ids = self._classified_ids()
        self.unclassified_ids = self._unclassified_ids()
        self.licit_ids = self._licit_ids()
        self.illicit_ids = self._illicit_ids()

    def visualize_distribution(self):
        groups = self.labels_df.groupby("class").count()
        plt.title("Classes distribution")
        plt.barh(['Licit', 'Illicit', 'Unknown'], groups['txId'].values, color=['green', 'red', 'grey'])

    def merge(self):
        df_merge = self.features_df.merge(self.labels_df, how='left', right_on="txId", left_on=0)
        df_merge = df_merge.sort_values(0).reset_index(drop=True)
        return df_merge

    def train_test_split(self, test_size=0.15):
        train_idx, valid_idx = train_test_split(self.classified_ids.values, test_size=test_size)
        return train_idx, valid_idx

    def pyg_dataset(self):
        dataset = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights,
            y=self.labels,
        )
        train_idx, valid_idx = self.train_test_split()
        dataset.train_idx = train_idx
        dataset.valid_idx = valid_idx
        dataset.test_idx = self.unclassified_ids

        return dataset

    def _licit_ids(self):
        node_features = self.merged_df.drop(['txId'], axis=1).copy()
        licit_ids = node_features['class'].loc[node_features['class'] == 0].index
        return licit_ids

    def _illicit_ids(self):
        node_features = self.merged_df.drop(['txId'], axis=1).copy()
        illicit_ids = node_features['class'].loc[node_features['class'] == 1].index
        return illicit_ids

    def _classified_ids(self):
        """
        Get the list of labeled ids
        """
        node_features = self.merged_df.drop(['txId'], axis=1).copy()
        unclassified_ids = node_features['class'].loc[node_features['class'] != 2].index
        return unclassified_ids

    def _unclassified_ids(self):
        """
        Get the list of unlabeled ids
        """
        node_features = self.merged_df.drop(['txId'], axis=1).copy()
        unclassified_ids = node_features['class'].loc[node_features['class'] == 2].index
        return unclassified_ids

    def _node_features(self):
        node_features = self.merged_df.drop(['txId'], axis=1).copy()
        node_features = node_features.drop(columns=[0, 1, "class"])
        node_features_t = torch.tensor(node_features.values, dtype=torch.double)

        return node_features_t

    def _edge_index(self):
        node_ids = self.merged_df[0].values
        ids_mapping = {y: x for x, y in enumerate(node_ids)}
        edges = self.edges_df.copy()
        edges.txId1 = edges.txId1.map(ids_mapping)  # get nodes idx1 from edges_df list and filtered data
        edges.txId2 = edges.txId2.map(ids_mapping)
        edges = edges.astype(int)

        edge_index = np.array(edges.values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        return edge_index

    def _edge_weights(self):
        weights = torch.tensor([1] * self.edge_index.shape[1], dtype=torch.double)
        return weights

    def _labels(self):
        labels = self.merged_df["class"].values
        labels_tensor = torch.tensor(labels, dtype=torch.double)
        return labels_tensor
