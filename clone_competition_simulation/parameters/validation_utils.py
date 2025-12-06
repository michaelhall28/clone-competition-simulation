from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator, BeforeValidator
import numpy as np
from numpy.typing import NDArray

from .algorithm_validation import Algorithm


def assign_config_settings(value, info):
    if value is not None:
        if isinstance(value, BaseModel):
            value = value.model_dump()
    else:
        value = {}
    if info.config['title'] == "SimulationRunSettings":
        value.update(info.data)  # Include any already-validated parameters as they might be needed
        config_settings = info.data.get("config_file_settings")
        value['config_file_settings'] = getattr(config_settings, info.field_name, {})
        value['tag'] = "Full"
    return value


class ValidationBase(BaseModel):
    algorithm: Algorithm
    validated: bool = False

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.validated:
            return self
        self._validate_model()
        self.validated = True
        return self

    def get_value_from_config(self, field_name: str):
        value = getattr(self, field_name)
        if value is None:
            return getattr(self.config_file_settings, field_name)
        return value


def convert_to_array(value):
    """
    Numbers and None are returned unchanged, otherwise
    convert anything "array like" to a numpy array.
    Args:
        value:

    Returns:

    """
    if value is None or isinstance(value, (int, float)):
        return value
    if not isinstance(value, np.ndarray):
        try:
            return np.array(value)
        except ValueError:
            return value
    return value


# Types
IntParameter = int | None
FloatParameter = float | None
ArrayParameter = Annotated[NDArray | None, BeforeValidator(convert_to_array)]
FloatOrArrayParameter = Annotated[NDArray | float | None, BeforeValidator(convert_to_array)]
IntOrArrayParameter = Annotated[NDArray | int | None, BeforeValidator(convert_to_array)]


# Fields
AlwaysValidateNoneField = Field(None, validate_default=True)
ValidationModelField = Field(None, validate_default=True, json_schema_extra={'config_validation': True})
