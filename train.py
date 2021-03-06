import datasets
import models
from torch.utils.data.dataloader import DataLoader
import option_parser
from tqdm import tqdm

def main():
    args = option_parser.get_args()

    amass_dataset = datasets.create_AMASSdataset(args)
    amass_loader = DataLoader(amass_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_worker,
                              drop_last = True)

    # Create DataSet
    #proxd_dataset = datasets.create_PROXDdataset(args)
    #data_loader = DataLoader(proxd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # Create Model
    model = models.create_GAN_model(args)
    for epoch in range(args.epoch_begin, args.epoch_num):
        print(epoch)
        for input_data in tqdm(amass_loader):
            model.set_input(input_data)
            model.optimize_parameters()


if __name__ == '__main__':
    main()