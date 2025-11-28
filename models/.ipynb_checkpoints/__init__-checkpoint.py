from .TRIT_Net import TRITNetEncoder,TRITNet

def _get_model_instance(name):
    try:
        return {
            "TRIT-Net-Encoder": TRITNetEncoder,
            "TRIT-Net": TRITNet
        }[name]
    except KeyError as e:
        raise ValueError(
            f"Model '{name}' not available"
        ) from e

def get_model(arch):
    name        = arch
    model       = _get_model_instance(name)
    model       = model()
   
    return model