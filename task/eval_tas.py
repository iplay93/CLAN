# trainer/eval_tas.py

def evaluate_model(model, classifier, test_list, test_label_list, args, configs, device, logger, experiment_log_dir, positive_aug):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from data_loader import Load_Dataset
    from model import model_evaluate
    from utils.metrics import _calc_metrics

    def to_tensor(x): return torch.tensor(x).cuda().cpu()
    test_list, test_label_list = map(to_tensor, [test_list, test_label_list])

    # Filter only known class
    known_class_idx = args.known_class_idx
    mask = np.isin(test_label_list, known_class_idx)
    test_list = test_list[mask]
    test_label_list = test_label_list[mask]

    test_dl = DataLoader(Load_Dataset(test_list, test_label_list, configs, args.training_mode, positive_aug),
                         batch_size=configs.batch_size, shuffle=True)

    logger.debug("Evaluate on the Test set")
    outs = model_evaluate(model, classifier, test_dl, device, args.training_mode, positive_aug)
    total_loss, total_acc, total_f1, auroc, pred_labels, true_labels = outs

    logger.debug(f'Test Loss : {total_loss:.4f} | Accuracy : {total_acc:.4f} | F1 : {total_f1:.4f} | AUROC : {auroc:.4f}')
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

    return total_loss, total_acc.item(), total_f1.item(), auroc.item()