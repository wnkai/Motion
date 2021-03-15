import datasets
import models
import option_parser

def main():
    args = option_parser.get_args()

    # Create Model
    model = models.create_GAN_model(args)
    model.load(17)

    proxd_dataset_test = datasets.create_PROXDdataset_noslice(args)

    for i in range(0,30):
        it, scence_name, filename = proxd_dataset_test.get_noslice(i)
        model.set_input(it)
        model.test(scence_name, filename)


if __name__ == '__main__':
    main()