from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import register_peft_method

from .saraconfig import SARAConfig
from .saralayer import Linear, SARALayer
from .saramodel import SARAModel


__all__ = ["Linear", "SARAConfig", "SARALayer", "SARAModel"]


register_peft_method(name="vera", config_cls=SaraConfig, model_cls=SaraModel, prefix="sara_lambda_")


def __getattr__(name):
    if (name == "Linear8bitLt") and is_bnb_available():
        from .bnb import Linear8bitLt

        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from .bnb import Linear4bit

        return Linear4bit

    raise AttributeError(f"module {__name__} has no attribute {name}")