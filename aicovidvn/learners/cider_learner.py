# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import re
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score

from aicovidvn.datasets import AICovidVNDataset
from aicovidvn.models.cider_model import CIdeRModel
from aicovidvn.utils import load_json, plot_roc_auc
from aicovidvn.utils import AddGaussianNoise, NoneTransform

class CIdeRLeaner():
    def __init__(
        self, 
        model: CIdeRModel=None,
        device: str=None
    ) -> None:
        super(CIdeRLeaner, self).__init__()

        self.model = model
        self.num_classes = self.model.num_classes
        
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
    def train(
        self, 
        root: str='./data/aivncovid-19',
        window_size: int=8,
        n_nfft: int=2048,
        sample_rate: int=48000,
        masking: bool=True,
        pitch_shift: bool=True,
        breathcough: bool=False,
        eval_type: str='maj_vote',
        noise: bool=True,
        batch_size: int=48,
        learning_rate: float=0.0001, 
        n_epochs: int=100,
        shuffle: bool=True,
        num_workers: int=4,
        view_model: bool=True,
        save_dir: str='./models',
        model_name: str='aivncovid',
        default_config_path: str='configs/default.json',
        **kwargs
    ):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise() if noise else NoneTransform(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # TODO: Create Dataset
        train_dataset = AICovidVNDataset(
            root=root, 
            mode='train',
            eval_type=eval_type, 
            transform=train_transform, 
            window_size=window_size,
            sample_rate=sample_rate,
            n_fft=n_nfft,
            masking=masking,
            pitch_shift=pitch_shift,
            breathcough=breathcough
        )
        valid_dataset = AICovidVNDataset(
            root=root, 
            mode='valid',
            eval_type=eval_type, 
            transform=train_transform, 
            window_size=window_size,
            sample_rate=sample_rate,
            n_fft=n_nfft,
            masking=masking,
            pitch_shift=pitch_shift,
            breathcough=breathcough
        )

        print(f"Length of Training dataset: {len(train_dataset)}")
        print(f"Length of Valid dataset: {len(valid_dataset)}")

        train_weight = self.make_sample_weights(train_dataset).to(self.device)
        valid_weight = self.make_sample_weights(valid_dataset).to(self.device)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=batch_size if eval_type != 'maj_vote' else 1, 
            shuffle=False,
            num_workers=num_workers
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of parameters is: {params}")

        if view_model:
            print(f"Model Info: ")
            print(self.model)

        best_score = 0

        for epoch in range(n_epochs):
            self._train(epoch, train_dataloader, optimizer, train_weight)
            f1_score, acc, recall, roc_auc = self.validate(epoch, valid_dataloader, valid_weight, eval_type)

            # TODO: Save model
            if f1_score > best_score:
                best_score = f1_score
                self.save_model(self.model, save_dir, model_name)
                print(f"Saved the best model !")

    @staticmethod
    def make_sample_weights(dataset):
        """
        Make the weights for each of the classes
        """
        data_df = dataset.data_df
        num_pos = data_df['label'].value_counts(normalize=True)[1]
        num_neg = data_df['label'].value_counts(normalize=True)[0]
        
        pos_weight = num_neg / num_pos
        print(f"Weight for Covid: {pos_weight}")
        weights = torch.Tensor([pos_weight])

        return weights

    def _train(
        self, 
        epoch, 
        train_dataloader, 
        optimizer, 
        train_weight,
        **kwargs
    ):
        self.model.train()
        train_dataloader = tqdm(train_dataloader, position=0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=train_weight)

        for i, (audio, label) in enumerate(train_dataloader):
            self.model.zero_grad()
            
            audio = audio.to(self.device)
            label = label.to(self.device)

            output = self.model(audio)
            loss = criterion(output, label.unsqueeze(1).float())
            loss.backward()

            # TODO: Get Accuracy
            logits = torch.sigmoid(output.detach())
            preds = np.where(logits.cpu().numpy()>0.5, 1, 0)
            score = f1_score(label.cpu().numpy(), preds)

            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            train_dataloader.set_description(
                (f'Epoch: {epoch + 1}; F1: {score:.5f}; Loss: {loss.item():.4f}')
            )

    def _eval(self, audio, label, criterion):
        audio = audio.to(self.device)
        label = label.to(self.device)

        output = self.model(audio)
        loss = criterion(output, label.unsqueeze(1).float())

        # TODO: Get Accuracy
        logits = torch.sigmoid(output).cpu().numpy()
        preds = np.where(logits > 0.5, 1, 0)

        return loss, preds, logits

    def validate(
        self, 
        epoch, 
        valid_dataloader, 
        valid_weight, 
        eval_type='random', 
        **kwargs
    ):
        self.model.eval()
        valid_dataloader = tqdm(valid_dataloader, position=0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=valid_weight)
        
        with torch.no_grad():
            ys = []
            y_hts = []
            logits_list = []
            losses = []

            for i, (audio, label) in enumerate(valid_dataloader):
                label = label.to(self.device)
                if eval_type == 'maj_vote':
                    loss, preds, _ = self._eval(audio, label, criterion)
                else:
                    clips = audio
                    clip_loss, clip_preds = 0, []
                    for audio in clips:
                        loss, preds, logits = self._eval(audio, label, criterion)
                        clip_loss += loss
                        clip_preds.append((preds, logits))

                    # TODO: Aggregate predicts and loss
                    loss = clip_loss / len(clips)
                    positive = np.count_nonzero([c[0] for c in clip_preds])
                    votes = {'1': positive, '0': len(clip_preds) - positive}

                    # If its a tie, use logits
                    if votes['1'] == votes['0']:
                        logits = (
                            sum([c[1] for c in clip_preds if c[0].item() == 0]), # Negative
                            sum([c[1] for c in clip_preds if c[0].item() == 1]), # Positive
                        )
                        preds = np.argmax(logits).reshape(1, 1)
                    else:
                        preds = np.array(int(max(votes.items(), key=lambda x: x[1])[0])).reshape(1,1)

                # For ROC-AUC
                average_logits = [c[1][0][0] for c in clip_preds]
                logits_list.append(np.mean(average_logits))

                y_hts.append(preds)
                ys.append(label.cpu().numpy())
                losses.append(loss.item())

            ys = np.concatenate(ys)
            y_hts = np.concatenate(y_hts)

            score = f1_score(ys, y_hts)
            acc = accuracy_score(ys, y_hts)
            recall = recall_score(ys, y_hts, average='macro')
            fpr, tpr, _ = roc_curve(ys, logits_list)
            roc_auc = auc(fpr, tpr)

            # TODO: Plot
            plot_roc_auc(fpr, tpr, roc_auc)

            valid_dataloader.set_description((
                f'Epoch: {epoch + 1}; Test-F1: {score:.5f}; Test-AUC {roc_auc:.5f}'
                f'Test-Loss: {sum(losses)/len(losses):.4f}'
            ))

        return score, acc, recall, roc_auc

    def save_model(
        self,
        model,
        save_dir: str='./models', 
        model_name: str='aivncovid'
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), path)
        with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
            f.write(str(model))
    