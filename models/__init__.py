from models.test_model import test_model,AE_model

def create_train_models(args):
    model = test_model(args)
    return model

def create_AE_model(args):
    model = AE_model(args)
    return model