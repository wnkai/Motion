import datasets
import models
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from rich.progress import track
import option_parser
import time

def main():
    args = option_parser.get_args()

    # Create DataSet
    dataset = datasets.create_dataset(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    # Create Model
    model = models.create_train_models(args)
    optimizer = optim.Adam(model.parameters())
    model = model.cuda()
    print(model)
    criterion_rec = torch.nn.MSELoss(reduce=True, size_average=True)
    for epoch in track(sequence = range(args.epoch_begin, args.epoch_num),
                       description ='Echo',):
        all_loss = 0
        for step, input_data in enumerate(data_loader):
            input_data = input_data.cuda()
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion_rec(input_data,output)
            loss.backward()
            optimizer.step()

            all_loss += loss * args.batch_size
        print(all_loss)


if __name__ == '__main__':
    main()