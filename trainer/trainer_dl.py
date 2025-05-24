import sys
sys.path.append("..")
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import *
import torch.fft as fft

# Custom utilities and modules
from models.loss import (
    NTXentLoss, NTXentLoss_poly, SupConLoss,
    get_similarity_matrix, NT_xent,
    get_similarity_two_matrix, NT_xent_TF
)
from data_preprocessing.augmentations import select_transformation
from utils.utils import plot_tsne_with_shift_labels

def Trainer(
    model, model_optimizer,
    classifier, classifier_optimizer,
    train_dl, device, logger,
    configs, experiment_log_dir, args,
    negative_list, negative_list_f,
    positive_list, positive_list_f
):
    """
    Trains the dual-encoder model and classifier with contrastive learning.
    
    Args:
        model: contrastive encoder model (e.g., TFC/TFT)
        model_optimizer: optimizer for the encoder
        classifier: classification head
        classifier_optimizer: optimizer for the classifier
        train_dl: dataloader for training
        device: torch device (cpu or cuda)
        logger: logging utility
        configs: configuration object
        experiment_log_dir: directory to save results
        args: command-line or argument object
        negative_list / negative_list_f: list of time/frequency domain negative transformations
        positive_list / positive_list_f: list of time/frequency domain positive transformations
    """
    logger.debug("Training started ...")

    # Use standard cross-entropy for auxiliary shift classification
    criterion = nn.CrossEntropyLoss()

    # Consistency checks on OOD configuration
    assert len(args.ood_score) == 1, "Only one OOD scoring method is supported."
    if args.ood_score[0] == 'simclr':
        assert args.K_shift == 1, "SimCLR setup expects K_shift = 1."
    else:
        assert args.K_shift > 1 and args.K_shift_f > 1, "Shifted classification requires K_shift > 1."

    # Training loop
    for epoch in range(1, configs.num_epoch + 1):
        if epoch % 50 == 0:
            logger.debug(f'\nEpoch: {epoch}\n')

        # Run one epoch of training
        train_loss, train_acc = model_train(
            epoch, logger, model, model_optimizer,
            classifier, classifier_optimizer, criterion,
            train_dl, configs, device, args,
            negative_list, negative_list_f,
            positive_list, positive_list_f
        )

        # Logging training progress
        if epoch % 50 == 0:
            logger.debug(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}\n')

    # Save final model checkpoint
    save_dir = os.path.join(experiment_log_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    chkpoint = {
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict()
    }
    torch.save(chkpoint, os.path.join(save_dir, 'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")
    return model