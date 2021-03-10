import datasets
import models
from torch.utils.data.dataloader import DataLoader
import option_parser
from tqdm import tqdm

def main():
    args = option_parser.get_args()

    # Create Model
    model = models.create_GAN_model(args)

    amass_dataset = datasets.create_AMASSdataset(args)
    amass_loader = DataLoader(amass_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, drop_last = True)

    prox_dataset = datasets.create_PROXDdataset(args)
    prox_loader = DataLoader(prox_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, drop_last=True)

    for epoch in range(args.epoch_begin, args.epoch_num):
        print(epoch)
        for input_data in tqdm(amass_loader):
            model.set_input(input_data)
            model.optimize_parameters()

        model.save()
        model.epoch_cnt = model.epoch_cnt + 1


if __name__ == '__main__':
    main()