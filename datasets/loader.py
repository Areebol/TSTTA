from torch.utils.data import DataLoader

from datasets.build import build_dataset


def construct_loader(cfg, split, batch_size=None):
    if split == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.TRAIN.SHUFFLE
        drop_last = cfg.TRAIN.DROP_LAST
    elif split == "val":
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = cfg.VAL.SHUFFLE
        drop_last = cfg.VAL.DROP_LAST
    elif split == "test":
        batch_size = cfg.TEST.BATCH_SIZE if batch_size is None else batch_size
        shuffle = cfg.TEST.SHUFFLE
        drop_last = cfg.TEST.DROP_LAST
    else:
        raise ValueError

    dataset = build_dataset(cfg, split)

    # 针对扰动数据集（测试阶段），使用按文件顺序与连续窗口分组的 batch sampler，
    # 确保一个 batch 内样本来自同一 CSV 且时间连续，避免一次性读取全部数据导致内存压力。
    if split == "test" and hasattr(dataset, "build_batch_sampler"):
        batch_sampler = dataset.build_batch_sampler(batch_size=batch_size, drop_last=drop_last)
    else:
        batch_sampler = None

    if batch_sampler is not None:
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
        )

    return loader


def get_train_dataloader(cfg):
    return construct_loader(cfg, "train")


def get_val_dataloader(cfg):
    return construct_loader(cfg, "val")


def get_test_dataloader(cfg, batch_size=None):
    return construct_loader(cfg, "test", batch_size=batch_size)
