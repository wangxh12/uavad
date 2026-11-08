from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ALFA_Forecast, Dataset_ALFA_Reconstruction

data_dict = {
    "ALFA": Dataset_ALFA_Forecast,
    'ALFA_Reconstruction': Dataset_ALFA_Reconstruction,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 1 if getattr(args, 'embed', '') == 'timeF' else 0

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=getattr(args, 'scale', 'standard'),
        timeenc=timeenc,
        freq=getattr(args, 'freq', 'h'),
        step=args.step,
        args=args,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=(flag == 'train'),
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f'{flag}: {len(data_set)} samples')
    return data_set, data_loader

