# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from typing import Union
from shutil import copyfile
from pandas import DataFrame
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score

from aicovidvn.datasets import AICovidVNDataset
from aicovidvn.models.cider_model import CIdeRModel
from aicovidvn.utils import ToFloatTensor, AddGaussianNoise, NoneTransform
from aicovidvn.utils import load_json, plot_roc_auc, print_free_style, save_json

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
        n_fft: int=2048,
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
        **kwargs
    ):
        train_transform = transforms.Compose([
            ToFloatTensor(),
            AddGaussianNoise() if noise else NoneTransform(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        valid_transform = transforms.Compose([
            ToFloatTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # TODO: Create Dataset
        train_dataset = AICovidVNDataset(
            root=root, 
            mode='train',
            transform=train_transform, 
            window_size=window_size,
            sample_rate=sample_rate,
            n_fft=n_fft,
            masking=masking,
            pitch_shift=pitch_shift,
            breathcough=breathcough
        )
        valid_dataset = AICovidVNDataset(
            root=root, 
            mode='valid',
            eval_type=eval_type, 
            transform=valid_transform, 
            sample_rate=sample_rate,
            window_size=window_size,
            n_fft=n_fft,
            breathcough=breathcough
        )
        
        print_free_style(message="Dataset Info")
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
            shuffle=shuffle,
            num_workers=num_workers
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if view_model:
            print_free_style(message="Model Info")
            print(f"Total number of parameters is: {params} \n")
            print(self.model)

        best_score = 0

        configs = {}
        configs['learning_rate'] = learning_rate
        configs['batch_size'] = batch_size
        configs['window_size'] = window_size
        configs['n_fft'] = n_fft
        configs['sample_rate'] = sample_rate
        configs['eval_type'] = eval_type
        configs['noise'] = noise
        configs['masking'] = masking
        configs['pitch_shift'] = pitch_shift
        configs['breathcough'] = breathcough
        configs['num_workers'] = num_workers

        save_json(save_dir='./configs', file='params.json', var=configs)

        for epoch in range(n_epochs):
            self._train(epoch, train_dataloader, optimizer, train_weight)
            f1_score, acc, recall, roc_auc = self.validate(epoch, valid_dataloader, valid_weight, eval_type)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs} Valid Evaluation: \n"
                        f"\tF1-score = {f1_score:.4f} | Acc = {acc:.4f} | Recall = {recall:.4f} | AUC = {roc_auc} \n"
            )

            # TODO: Save model
            if roc_auc > best_score:
                best_score = roc_auc
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

        for i, (path, audio, label) in enumerate(train_dataloader):
            self.model.zero_grad()
            
            audio = audio.to(self.device)
            label = label.to(self.device)

            output = self.model(audio)
            loss = criterion(output, label.unsqueeze(1).float())
            loss.backward()

            # TODO: Get Accuracy
            logits = torch.sigmoid(output.detach())
            preds = np.where(logits.cpu().numpy()>0.5, 1, 0)
            acc = accuracy_score(label.cpu().numpy(), preds)

            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            train_dataloader.set_description(
                (f'Epoch {epoch + 1}: Loss = {loss.item():.4f} | Acc = {acc:.4f}')
            )

    def _eval(self, audio, label, criterion=None):
        audio = audio.to(self.device)
        label = label.to(self.device)

        output = self.model(audio)
        loss = None
        if criterion:
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

            for i, (path, audio, label) in enumerate(valid_dataloader):
                label = label.to(self.device)
                if eval_type != 'maj_vote':
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
                        preds = np.array(int(max(votes.items(), key=lambda x: x[1])[0])).reshape(1, 1)

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
                f'Epoch {epoch + 1}: Test-F1: {score:.4f}; Test-AUC {roc_auc:.4f}'
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

    def load_model(
        self,
        model_path: str=None,
    ):
        # Check the model file exists
        if not os.path.isfile(model_path):
            raise ValueError(f"The model file `{model_path}` is not exists or broken! ")

        self.model.load_state_dict(torch.load(model_path))
    
    def batch_inference(
        self,
        input: Union[str, DataFrame],
        save_dir: str='./output', 
        file_name: str='output.csv'
    ):
        self.model.eval()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if isinstance(input, DataFrame):
            input.to_csv(os.path.join(save_dir, 'inference.csv'), encoding='utf-8', index=False)
        else:
            # Check the model file exists
            if not os.path.isfile(input):
                raise ValueError(f"The input file `{input}` is not exists or broken! ")
            
            dsc = os.path.join(save_dir, 'inference.csv')
            copyfile(input, dsc)

        cfg = load_json(file='./configs/params.json')

        test_transform = transforms.Compose([
            ToFloatTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # TO DO: Create dataset
        test_dataset = AICovidVNDataset(
            root=save_dir,
            mode='inference',
            transform=test_transform,
            eval_type=cfg['eval_type'],
            window_size=cfg['window_size'],
            sample_rate=cfg['sample_rate'],
            n_fft=cfg['n_fft'],
            breathcough=cfg['breathcough']
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg['batch_size'] if cfg['eval_type'] != 'maj_vote' else 1,
            shuffle=False,
            num_workers=cfg['num_workers']
        )

        with torch.no_grad():
            paths = []
            assessment_results = []
            for (path, audio, label) in tqdm(test_dataloader):
                if cfg['eval_type'] != 'maj_vote':
                    _, preds, logits = self._eval(audio=audio, label=label)
                else:
                    clips = audio
                    clip_preds = []
                    for audio in clips:
                        _, preds, logits = self._eval(audio=audio, label=label)
                        clip_preds.append((preds, logits))

                    # TODO: Aggregate predicts and loss
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
                        preds = np.array(int(max(votes.items(), key=lambda x: x[1])[0])).reshape(1, 1)
                    
                average_logits = [c[1][0][0] for c in clip_preds]
                logits = np.mean(average_logits)
                
                paths.append(path[0])
                assessment_results.append(logits)

        # TODO: Get uuid
        uuids = [path.split('/')[-1].split('.')[0] for path in paths]

        outdf = pd.DataFrame({
            'path': paths,
            'uuid': uuids,
            'assessment_result': assessment_results
        })
        
        outdf.to_csv(os.path.join(save_dir, file_name), encoding='utf-8', index=False)
        
        return outdf
