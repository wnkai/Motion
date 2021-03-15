from models.architecture import GAN_model

def create_GAN_model(args, dataset):
    model = GAN_model(args, dataset)
    return model