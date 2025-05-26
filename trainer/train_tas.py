
import os
import torch


def prepare_environment(args):
    """
    Configure device, dataset settings, result file paths, and import dataset-specific config.

    Args:
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        dict: A dictionary containing device, dataset class list, output paths, configs
    """
    # Device setup
    device = torch.device(args.device)
    
    # Static values
    method = 'Test OOD-ness'
    pos_ths = 0.6
    neg_ths = args.neg_ths

    # Output file paths based on training mode
    if args.training_mode == 'T':
        suffix = '_T_' + str(neg_ths)
    elif args.training_mode == 'F':
        suffix = '_F_' + str(neg_ths)
    else:
        raise ValueError("Invalid training_mode. Use 'T' or 'F'.")

    store_path = f'result_files/final_ood_{args.selected_dataset}{suffix}.xlsx'
    store_path_2 = f'result_files/Summary_ood_{args.selected_dataset}{suffix}.xlsx'

    # Ensure log directory exists
    os.makedirs(args.logs_save_dir, exist_ok=True)

    # Dynamically import dataset-specific config
    exec(f'from config_files.{args.selected_dataset}_Configs import Config as Configs')
    configs = Configs()

    # Dataset-specific settings
    class_num = [-1]  # Default
    if args.selected_dataset == 'lapras':
        args.timespan = 10000
        class_num = [0, 1, 2, 3, -1]

    elif args.selected_dataset == 'casas':
        class_num = list(range(14)) + [-1]
        args.aug_wise = 'Temporal2'

    elif args.selected_dataset == 'opportunity':
        args.timespan = 1000
        class_num = [0, 1, 2, 3, 4, -1]

    elif args.selected_dataset == 'aras_a':
        args.timespan = 1000
        class_num = list(range(16)) + [-1]

    elif args.selected_dataset == 'openpack':
        args.timespan = 100
        class_num = list(range(10)) + [-1]

    elif args.selected_dataset == 'pamap':
        args.timespan = 0
        args.aug_wise = 'None'
        class_num = list(range(12)) + [-1]

    else:
        raise ValueError(f"Unsupported dataset: {args.selected_dataset}")

    return {
        "device": device,
        "class_num": class_num,
        "store_path": store_path,
        "store_path_2": store_path_2,
        "configs": configs,
        "method": method,
        "pos_ths": pos_ths,
        "neg_ths": neg_ths
    }


# ================== Example Main Entrypoint ==================
if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser() ... define args here or import from your config
    args = parser.parse_args()
    
    env = prepare_environment(args)
    
    # Use env values:
    device = env["device"]
    class_num = env["class_num"]
    store_path = env["store_path"]
    store_path_2 = env["store_path_2"]
    configs = env["configs"]
    method = env["method"]
    pos_ths = env["pos_ths"]
    neg_ths = env["neg_ths"]