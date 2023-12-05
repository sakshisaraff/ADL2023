def create_spectrogram:

    default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"

    path_annotations_train = Path("annotations/train_labels.pkl")
    path_annotations_val = Path("annotations/val_labels.pkl")
    path_annotations_test = Path("annotations/test_labels.pkl")
    train_dataset = dataset.MagnaTagATune(path_annotations_train, Path("samples/"))
    val_dataset = dataset.MagnaTagATune(path_annotations_val, Path("samples/"))
    test_dataset = dataset.MagnaTagATune(path_annotations_test, Path("samples/"))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    train_auc = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    return train_loader, train_auc, val_loader, test_loader