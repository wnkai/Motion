import datasets
import models
from torch.utils.data.dataloader import DataLoader
import option_parser
from tqdm import tqdm

def main():
    args = option_parser.get_args()

    # Create Model
    model = models.create_GAN_model(args)

    proxd_dataset_test = datasets.create_PROXDdataset_noslice(args)

    proxd_dataset = datasets.create_PROXDdataset(args)
    proxd_loader = DataLoader(proxd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    amass_dataset = datasets.create_AMASSdataset(args)
    amass_loader = DataLoader(amass_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_worker,
                              drop_last = True)


    for epoch in range(args.epoch_begin, args.epoch_num):
        print(epoch)
        for input_data in tqdm(amass_loader):
            model.set_input(input_data)
            model.optimize_parameters()

        if epoch != 0 and epoch % 5 == 0:
            it, scence_name = proxd_dataset_test.get_noslice(0)
            model.set_input(it)
            model.test(scence_name)


if __name__ == '__main__':
    main()