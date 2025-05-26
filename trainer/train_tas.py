import os
import argparse
import torch

def parse_arguments():
    """Parse command-line arguments for Novelty Detection configuration."""
    parser = argparse.ArgumentParser(description="Argument parser for Novelty Detection")

    home_dir = os.getcwd()

    # Experiment info
    parser.add_argument('--experiment_description', default='Exp1', type=str)
    parser.add_argument('--run_description', default='run1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--training_mode', default='T', choices=['T', 'F'],
                        help='Data augmentation type: "T" for Temporal, "F" for Frequency')
    parser.add_argument('--version', default='CL', type=str, choices=['CL', 'ND'])

    # Dataset
    parser.add_argument('--selected_dataset', default='casas', type=str,
                        choices=['lapras', 'casas', 'opportunity', 'aras_a', 'openpack', 'pamap'])
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str)
    parser.add_argument('--home_path', default=home_dir, type=str)

    # Device
    parser.add_argument('--device', default='cuda', type=str)

    # Sequence parameters
    parser.add_argument('--padding', default='mean', choices=['no', 'max', 'mean'])
    parser.add_argument('--timespan', type=int, default=1000)
    parser.add_argument('--min_seq', type=int, default=10)
    parser.add_argument('--min_samples', type=int, default=20)
    parser.add_argument('--one_class_idx', type=int, default=-1)

    # Novelty Detection (ND) parameters
    parser.add_argument('--nd_score', default=['norm_mean'], nargs='+', type=str,
                        help='Score function(s) for Novelty Detection')
    parser.add_argument('--nd_samples', default=1, type=int,
                        help='Number of samples to compute ND score')
    parser.add_argument('--print_nd_score', action='store_true',
                        help='Print ND score quantiles')
    parser.add_argument('--nd_layer', default=['simclr', 'shift'], nargs='+',
                        choices=['penultimate', 'simclr', 'shift'],
                        help='Feature layer(s) to use for ND scoring')

    # Augmentation
    parser.add_argument('--aug_method', default='AddNoise', type=str)
    parser.add_argument('--aug_wise', default='Temporal', choices=['None', 'Temporal', 'Sensor'])

    # Data split
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--overlapped_ratio', type=int, default=50)

    # Training settings
    parser.add_argument('--loss', default='SupCon', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--neg_ths', type=float, default=0.9)

    # Logging
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=50)

    return parser.parse_args()

def setup_environment(args):
    """Prepare device, output paths, dataset-specific config, and class index list."""
    torch.cuda.empty_cache()
    device = torch.device(args.device)

    # Output file paths based on training mode
    suffix = f"_{args.training_mode}_{args.neg_ths}"
    store_path = f"result_files/final_nd_{args.selected_dataset}{suffix}.xlsx"
    store_path_2 = f"result_files/Summary_nd_{args.selected_dataset}{suffix}.xlsx"

    # Ensure log directory exists
    os.makedirs(args.logs_save_dir, exist_ok=True)

    # Load dataset-specific config
    exec(f'from config_files.{args.selected_dataset}_Configs import Config as Configs')
    configs = Configs()

    # Dataset-specific settings
    dataset_classes = {
        'lapras': (10000, [0, 1, 2, 3, -1]),
        'casas': (args.timespan, list(range(14)) + [-1]),
        'opportunity': (1000, [0, 1, 2, 3, 4, -1]),
        'aras_a': (1000, list(range(16)) + [-1]),
        'openpack': (100, list(range(10)) + [-1]),
        'pamap': (0, list(range(12)) + [-1])
    }

    args.timespan, class_num = dataset_classes[args.selected_dataset]

    # Override augmentation strategy for specific datasets
    if args.selected_dataset == 'casas':
        args.aug_wise = 'Temporal2'
    elif args.selected_dataset == 'pamap':
        args.aug_wise = 'None'

    return device, store_path, store_path_2, configs, class_num

def main():
    args = parse_arguments()
    device, store_path, store_path_2, configs, class_num = setup_environment(args)

    # Example usage for verification
    print(f"🚀 Running Novelty Detection experiment: {args.experiment_description} - {args.run_description}")
    print(f"📂 Dataset: {args.selected_dataset} | Mode: {args.training_mode}")
    print(f"🖥️ Device: {device}")
    print(f"📁 Output: {store_path}, {store_path_2}")
    print(f"📊 Classes: {class_num}")
    print(f"🔧 Augmentation: {args.aug_wise} | Score method(s): {args.nd_score}")

if __name__ == "__main__":
    main()