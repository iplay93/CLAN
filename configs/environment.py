# config/environment.py
import os
import torch

def setup_environment(args):
    torch.cuda.empty_cache()
    device = torch.device(args.device)

    suffix = f"_{args.training_mode}_{args.neg_ths}"
    store_path = f"result_files/final_nd_{args.selected_dataset}{suffix}.xlsx"
    store_path_2 = f"result_files/Summary_nd_{args.selected_dataset}{suffix}.xlsx"

    os.makedirs(args.logs_save_dir, exist_ok=True)

    exec(f'from config_files.{args.selected_dataset}_Configs import Config as Configs')
    configs = Configs()

    dataset_classes = {
        'lapras': (10000, [0, 1, 2, 3, -1]),
        'casas': (args.timespan, list(range(14)) + [-1]),
        'opportunity': (1000, [0, 1, 2, 3, 4, -1]),
        'aras_a': (1000, list(range(16)) + [-1]),
        'openpack': (100, list(range(10)) + [-1]),
        'pamap': (0, list(range(12)) + [-1]),
    }

    args.timespan, class_num = dataset_classes[args.selected_dataset]

    if args.selected_dataset == 'casas':
        args.aug_wise = 'Temporal2'
    elif args.selected_dataset == 'pamap':
        args.aug_wise = 'None'

    return device, store_path, store_path_2, configs, class_num