from models.architecture import GAN_model

def create_GAN_model(args):
    model = GAN_model(args)
    return model