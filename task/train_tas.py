# main.py for TAS construction

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up environment: device, paths, configs, and class indices
    device, store_path, store_path_2, configs, class_num = setup_environment(args)

    # Load dataset and corresponding labels
    _, datalist, labellist = loading_data(args.selected_dataset, args)

    # Display key experiment information for traceability
    print(f"[INFO] Experiment: {args.experiment_description} | Run: {args.run_description}")
    print(f"[INFO] Dataset: {args.selected_dataset} | Mode: {args.training_mode}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output files: {store_path}, {store_path_2}")
    print(f"[INFO] Classes: {class_num}")
    print(f"[INFO] ND Score: {args.nd_score}")

    # Set the target class and data augmentation for this experiment
    args.one_class_idx = class_num[0]  # Example: use the first class for one-class training
    positive_aug = "AddNoise"         # Example augmentation method

    # Set up logging directory and logger
    experiment_log_dir = os.path.join(
        args.logs_save_dir, args.experiment_description,
        args.run_description, f"{args.training_mode}_seed_{args.seed}"
    )
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Create a timestamped log file for this experiment run
    log_path = os.path.join(
        experiment_log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = _logger(log_path)

    # Training Phase
    trainer, model, classifier, known_class_idx = train_model(
        args, configs, class_num, datalist, labellist,
        args.selected_dataset, device, positive_aug, logger, experiment_log_dir
    )

    # Store known class indices for use during evaluation
    args.known_class_idx = known_class_idx

    # Evaluation Phase
    _, _, _, _ = evaluate_model(
        model, classifier, datalist, labellist, args,
        configs, device, logger, experiment_log_dir, positive_aug
    )

if __name__ == "__main__":
    main()
