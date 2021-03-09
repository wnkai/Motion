import datasets
import models
import option_parser

def main():
    args = option_parser.get_args()

    # Create Model
    model = models.create_GAN_model(args)
    model.load(1)

    proxd_dataset_test = datasets.create_PROXDdataset_noslice(args)

    it, scence_name = proxd_dataset_test.get_noslice(0)
    model.set_input(it)
    model.test(scence_name)


if __name__ == '__main__':
    main()