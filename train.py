import datasets
import models
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from rich.progress import track
import option_parser

def main():
    args = option_parser.get_args()

    #amass_dataset = datasets.create_AMASSdataset(args)
    #amass_loader = DataLoader(amass_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # Create DataSet
    proxd_dataset = datasets.create_PROXDdataset(args)
    data_loader = DataLoader(proxd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # Create Model
    model = models.create_AE_model(args)
    for epoch in track(sequence = range(args.epoch_begin, args.epoch_num),
                       description ='Echo',):
        for step, input_data in enumerate(data_loader):
            model.set_input(input_data)
            model.optimize_parameters()


if __name__ == '__main__':
    main()