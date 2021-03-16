import datasets
import models
from torch.utils.data.dataloader import DataLoader
import option_parser
from tqdm import tqdm

def main():
    args = option_parser.get_args()

    mix_dataset = datasets.create_MixedData(args)
    mix_loader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, drop_last=True)

    prox_dataset = datasets.create_PROXDdataset_noslice(args)


    # Create Model
    model = models.create_GAN_model(args, mix_dataset)

    for epoch in range(args.epoch_begin, args.epoch_num):
        print(epoch)
        for input_data in tqdm(mix_loader):
            model.set_input(input_data)
            model.optimize_parameters()

        prox_data, scence_name, name = prox_dataset.get_noslice(0)
        model.compute_test_result(prox_data)

        model.save()
        model.epoch_cnt += 1


if __name__ == '__main__':
    main()