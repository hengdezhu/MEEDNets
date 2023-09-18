import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loader import get_test_loader_from_mat
from models import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Customisation
# penalties = [0]
# ratios = [0.1]
# datasets = ['sarscov2-ctscan', 'brain-tumor']

penalties = [0]
ratios = [0.1]
datasets = ['covid_pneumonia_normal']
classes = {
    'ALL':4,
    'brain-tumor':3,
    'sarscov2-ctscan':2,
    'lung-cancer':3,
    'chest_xray':2,
    'tuberculosis':2,
    'ct_kidney':4,
    'covid_pneumonia_normal':3,
    'xray_dataset_covid19':2
}

num_folds = 5

def load_model(model_path, num_classes, ancestor=False):
    checkpoint = torch.load(model_path)
    if ancestor:
        model = densenet.densenet121(weights=None)
    else:
        cfg = checkpoint['cfg']
        model = densenet.densenet121(weights=None, cfg=cfg)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def cal_metrics(y_true, y_pred, average='macro'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    return [accuracy, precision, recall, f1]

def validation(model, test_loader):
    model.cuda()
    model.eval()

    preds = []
    targets = []

    with tqdm(total=len(test_loader.dataset)) as pbar:
        pbar.set_description('Evaluation')
        
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            preds = np.append(preds, pred.data.cpu().numpy())
            targets = np.append(targets, target.data.cpu().numpy())

            batch_size = data.shape[0]
            pbar.update(batch_size)
    
    return cal_metrics(targets, preds)

SAVE_PATH = 'table_results/penalty_{}/ratio_{}/{}'
BATCH_SIZE = 8

kwargs = {'num_workers':1, 'pin_memory':False}

for dataset in datasets:
    num_classes = classes[dataset]
    for penalty in penalties:
        for ratio in ratios:
            models_dir = 'trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}'
            min_geneartions = 15
            # Check the maximum number of generation
            for fold_num in range(1, num_folds+1):
                num = len(os.listdir(models_dir.format(penalty, ratio, dataset, fold_num)))
                if min_geneartions >= num:
                    min_geneartions = num

            five_fold_results = []

            # 'table_results/penalty_{}/ratio_{}/{}'
            save_dir = SAVE_PATH.format(penalty, ratio, dataset)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Evaluate each geneartion in each Fold
            for fold_num in range(1, num_folds+1):
                single_fold_result = {}  # Save the resutls obtained by all generations
                metrics = []

                # Load the test data of that fold
                data_path = '../data/mat/{}_Fold_{}.mat'.format(dataset, fold_num)
                test_loader = get_test_loader_from_mat(data_path, BATCH_SIZE, fold_num=str(fold_num), **kwargs)

                # Inference of the ancestor network
                model_path = 'trained_models/penalty_{}/ancestor/{}/Fold_{}/pruned_checkpoint_0.pth.tar'.format(penalty, dataset, fold_num)
                model = load_model(model_path, num_classes, True)
                metrics = validation(model, test_loader)
                single_fold_result['Ancestor'] = metrics
                
                # Inference of each descendant network
                for generation in range(1, min_geneartions+1):
                    model_path = 'trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}/pruned_checkpoint_{}.pth.tar'.format(penalty, ratio, dataset, fold_num, generation)
                    model = load_model(model_path, num_classes)

                    metrics = validation(model, test_loader)

                    gen_name = ''

                    if generation == 1:
                        gen_name = '1st Generation'
                    elif generation == 2:
                        gen_name = '2nd Generation'
                    elif generation == 3:
                        gen_name = '3rd Generation'
                    else:
                        gen_name = '{}th Generation'.format(generation)

                    single_fold_result[gen_name] = metrics

                # Generate table for each fold
                df_single_fold = pd.DataFrame.from_dict(single_fold_result, orient='index', columns=['ACC', 'PRE', 'SEN', 'F1s']) * 100.

                with open(os.path.join(save_dir, f'Fold_{fold_num}.txt'), 'w') as file:
                    file.writelines(df_single_fold.round(2).style.to_latex())

                five_fold_results.append(df_single_fold)

            # Generate the overall result on cross-validation
            mean_df = (
                # combine dataframes into a single dataframe
                pd.concat(five_fold_results)
                .reset_index()
                # group by the row within the original dataframe
                .groupby("index", sort=False)
                # calculate the mean
                .mean()
            ).round(2)
            mean_df.index.name = None

            std_df = (
                # combine dataframes into a single dataframe
                pd.concat(five_fold_results)
                .reset_index()
                # group by the row within the original dataframe
                .groupby("index", sort=False)
                # calculate the std
                .std()
            ).round(2)
            std_df.index.name = None
            
            overall_df = mean_df.astype(str) + '±' + std_df.astype(str)

            with open(os.path.join(save_dir, 'Overall.txt'), 'w') as file:
                file.writelines(overall_df.style.to_latex())