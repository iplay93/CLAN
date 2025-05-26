# trainer/train_tas.py

from datetime import datetime
import os, random, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.logger import _logger
from utils.metrics import _calc_metrics

import random, math

def train_model(args, configs, class_num, datalist, labellist, data_type, device, positive_aug, logger, experiment_log_dir):
    from model import TFC_one, target_classifier, Trainer
    from data_loader import Load_Dataset
    from utils.label_util import count_label_labellist

    # Split dataset
    from sklearn.model_selection import train_test_split
    seed = args.seed
    train_list, test_list, train_label_list, test_label_list = train_test_split(
        datalist, labellist, test_size=args.test_ratio, stratify=labellist, random_state=seed)
    train_list, valid_list, train_label_list, valid_label_list = train_test_split(
        train_list, train_label_list, test_size=args.valid_ratio, stratify=train_label_list, random_state=seed)

    # Known class filtering
    import numpy as np, torch
    def to_tensor(x): return torch.tensor(x).cuda().cpu()
    train_list, valid_list, test_list = map(to_tensor, [train_list, valid_list, test_list])
    train_label_list, valid_label_list, test_label_list = map(to_tensor, [train_label_list, valid_label_list, test_label_list])

    exist_labels, _ = count_label_labellist(train_label_list)
    if args.one_class_idx != -1:
        known_class_idx = [args.one_class_idx]
    else:

        sup_class_idx = list(exist_labels)
        known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))

    def filter_known(data, label):
        mask = np.isin(label, known_class_idx)
        return data[mask], label[mask]

    train_list, train_label_list = filter_known(train_list, train_label_list)
    valid_list, valid_label_list = filter_known(valid_list, valid_label_list)

    # Dataloaders
    from torch.utils.data import DataLoader
    train_dl = DataLoader(Load_Dataset(train_list, train_label_list, configs, args.training_mode, positive_aug),
                          batch_size=configs.batch_size, shuffle=True)
    valid_dl = DataLoader(Load_Dataset(valid_list, valid_label_list, configs, args.training_mode, positive_aug),
                          batch_size=configs.batch_size, shuffle=True)

    # Model init
    model = TFC_one(configs, args).to(device)
    classifier = target_classifier(configs).to(device)

    model_opt = torch.optim.Adam(model.parameters(), lr=configs.lr,
                                 betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    clf_opt = torch.optim.Adam(classifier.parameters(), lr=configs.lr,
                               betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    trainer = Trainer(model, model_opt, classifier, clf_opt,
                      train_dl, valid_dl, None, device, logger,
                      configs, experiment_log_dir, args.training_mode, positive_aug)

    return trainer, model, classifier, known_class_idx