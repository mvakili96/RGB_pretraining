
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop, AdamW
from .lars import LARS

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
    "lars" : LARS, 
    "adamw" : AdamW,
}


def get_optimizer(name):
    opt_name = name
    if opt_name is None or opt_name not in key2opt:
        raise NotImplementedError("Optimizer {} not implemented".format(opt_name))
    else:            
        return key2opt[opt_name]