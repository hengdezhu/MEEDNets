import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import *
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

penalties = [0]
ratios = [0.1]
datasets = ['brain-tumor', 'sarscov2-ctscan']
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
SAVE_PATH = 'model_complexity/ensemble_networks/models_3/penalty_{}/ratio_{}/{}'

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

for dataset in datasets:
    num_classes = classes[dataset]
    for penalty in penalties:
        for ratio in ratios:
            models_dir = 'trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}'
            min_geneartions = 10
            # Check the maximum number of generation
            for fold_num in range(1, num_folds+1):
                num = len(os.listdir(models_dir.format(penalty, ratio, dataset, fold_num)))
                if min_geneartions >= num:
                    min_geneartions = num

            five_fold_results = []

            # 'model_complexity/penalty_{}/ratio_{}/{}'
            save_dir = SAVE_PATH.format(penalty, ratio, dataset)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Evaluate each geneartion in each Fold
            for fold_num in range(1, num_folds+1):
                single_fold_result = {}  # Save the resutls obtained by all generations
                metrics = []

                # For ensemble models. Comment this line for single network complexity analysis
                ens_macs, ens_params = 0, 0

                # Inference of the ancestor network
                model_path = 'trained_models/penalty_{}/ancestor/{}/Fold_{}/pruned_checkpoint_0.pth.tar'.format(penalty, dataset, fold_num)
                model = load_model(model_path, num_classes, True)
                macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False, verbose=False)
                metrics = [macs/1000000000, params/1000000]
                single_fold_result['Ancestor'] = metrics
                
                # Inference of each descendant network
                for generation in range(8, min_geneartions+1):
                    model_path = 'trained_models/penalty_{}/trained_models/ratio_{}/{}/Fold_{}/pruned_checkpoint_{}.pth.tar'.format(penalty, ratio, dataset, fold_num, generation)
                    model = load_model(model_path, num_classes)

                    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=False, print_per_layer_stat=False, verbose=False)
                    metrics = [macs/1000000000, params/1000000]

                    # For ensemble models. Comment these lines for single network complexity analysis
                    ens_macs += macs
                    ens_params += params

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

                # For ensemble models. Comment this line for single network complexity analysis
                single_fold_result['Ens'] = [ens_macs/1000000000, ens_params/1000000]

                # Generate table for each fold
                df_single_fold = pd.DataFrame.from_dict(single_fold_result, orient='index', columns=['MACs', 'Param'])

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