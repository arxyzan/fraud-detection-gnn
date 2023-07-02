import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import networkx as nx
import matplotlib.pyplot as plt

from .models import GAT, GCN, GIN
from .datasets import EllipticDataset

models_map = {
    "gcn": GCN,
    "gat": GAT,
    "gin": GIN
}

datasets_map = {
    "elliptic": EllipticDataset
}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset = datasets_map[self.config.train.dataset](config.dataset).pyg_dataset().to(config.train.device)
        self.config.model.input_dim = self.dataset.num_node_features

        self.model = models_map[self.config.train.model](config.model).double()
        self.model.to(config.train.device)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        self.tensorboard = SummaryWriter()

        self.metrics_outputs = {
            "train": {
                "accuracy": [],
                "f1_micro": [],
                "f1_macro": [],
                "recall": [],
                "precision": [],
                "confusion_matrix": [],
            },
            "eval": {
                "accuracy": [],
                "f1_micro": [],
                "f1_macro": [],
                "recall": [],
                "precision": [],
                "confusion_matrix": [],
            }
        }

    def compute_metrics(self, preds, labels, mode, threshold=0.5):
        preds = preds > threshold

        accuracy = accuracy_score(labels, preds)
        f1_micro = f1_score(labels, preds, average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=1)
        cm = confusion_matrix(labels, preds)

        self.metrics_outputs[mode]["accuracy"].append(accuracy)
        self.metrics_outputs[mode]["f1_micro"].append(f1_micro)
        self.metrics_outputs[mode]["f1_macro"].append(f1_macro)
        self.metrics_outputs[mode]["recall"].append(recall)
        self.metrics_outputs[mode]["precision"].append(precision)
        self.metrics_outputs[mode]["confusion_matrix"].append(cm)

        return {
            "accuracy": accuracy,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "recall": recall,
            "precision": precision,
        }

    def train(self):
        for epoch in range(1, self.config.train.num_epochs):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(self.dataset)
            outputs = outputs.reshape((self.dataset.x.shape[0]))
            loss = self.criterion(outputs[self.dataset.train_idx], self.dataset.y[self.dataset.train_idx])

            loss.backward()
            self.optimizer.step()

            labels = self.dataset.y.detach().cpu().numpy()[self.dataset.train_idx]
            preds = outputs.detach().cpu().numpy()[self.dataset.train_idx]
            train_results = self.compute_metrics(preds, labels, mode="train", threshold=0.5)

            # Evaluation step
            self.model.eval()
            labels = self.dataset.y.detach().cpu().numpy()[self.dataset.valid_idx]
            preds = outputs.detach().cpu().numpy()[self.dataset.valid_idx]
            eval_results = self.compute_metrics(preds, labels, mode="eval", threshold=0.5)

            # Print results
            if not epoch % self.config.train.print_freq:
                print(f"epoch: {epoch}: \n"
                      f"Train results: {train_results}\n"
                      f"Evaluation results: {eval_results}")

            # Tensorboard
            for metric, value in train_results.items():
                self.tensorboard.add_scalar(f"train/{metric}", value, epoch)
            for metric, value in eval_results.items():
                self.tensorboard.add_scalar(f"eval/{metric}", value, epoch)

    def test(self, dataset=None, labeled_only=False, threshold=0.5):

        dataset = dataset or self.dataset

        self.model.eval()
        outputs = self.model(dataset)
        outputs = outputs.reshape((dataset.x.shape[0]))

        if labeled_only:
            preds = outputs.detach().cpu().numpy()
        else:
            preds = outputs.detach().cpu().numpy()[dataset.test_idx]

        pred_labels = preds > threshold

        return preds, pred_labels

    def save(self, file_name):
        file_name = f"{file_name}.pt" if ".pt" not in file_name else file_name
        save_path = os.path.join(self.config.train.save_dir, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def visualize(self, dataset: EllipticDataset, time_step: int, save_to=None):
        pred_scores, pred_labels = self.test(dataset.pyg_dataset().to(self.config.train.device), labeled_only=True)
        node_list = dataset.merged_df.index[dataset.merged_df.loc[:, 1] == time_step].tolist()

        edge_tuples = []
        for row in dataset.edge_index.view(-1, 2).cpu().numpy():
            if (row[0] in node_list) | (row[1] in node_list):
                edge_tuples.append(tuple(row))

        # Fetch predicted results for that time period
        node_color = []
        for node_id in node_list:
            if node_id in dataset.illicit_ids:
                label = "red"  # fraud
            elif node_id in dataset.licit_ids:
                label = "green"  # not fraud
            else:
                if pred_labels[node_id]:
                    label = "orange"  # Predicted fraud
                else:
                    label = "blue"  # Not fraud predicted

            node_color.append(label)

        # Setup networkx graph
        G = nx.Graph()
        G.add_edges_from(edge_tuples)

        # Plot the graph
        plt.figure(3, figsize=(16, 16))
        plt.title("Time period:" + str(time_step))
        nx.draw_networkx(G, nodelist=node_list, node_color=node_color, node_size=6, with_labels=False)
        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            plt.savefig(save_to)
            print(f"Graph visualization saved to `{save_to}`")

