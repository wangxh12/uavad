from torch.utils.data import DataLoader

from data_provider.data_loader import (
    ALFAForecast,
    ALFARecon,
    ALFASegForecast,
    ALFASegRecon,
)

data_dict = {
    "ALFA": ALFARecon,
    "ALFA_Forecast": ALFAForecast,
    "ALFA_Seg": ALFASegRecon,
    "ALFA_Seg_Forecast": ALFASegForecast,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 1 if getattr(args, "embed", "") == "timeF" else 0

    if args.data == "PretrainSegLoader":
        data_set = Data(
            root_path=args.root_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=args.stride,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=(flag == "train"),
            num_workers=args.num_workers,
            drop_last=False,
        )
        print(f"{flag}: {len(data_set)} samples")
        return data_set, data_loader

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        stride=args.stride,
        scaler=args.scaler,
        args=args,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=(flag == "train"),
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"{flag}: {len(data_set)} samples")
    return data_set, data_loader
