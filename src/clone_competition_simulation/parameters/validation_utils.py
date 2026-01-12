from functools import partial
from typing import Annotated, Self, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator, BeforeValidator

from .algorithm_validation import Algorithm


def assign_config_settings(value, info):
    if value is not None:
        if isinstance(value, BaseModel):
            value = value.model_dump()
    else:
        value = {}
    if info.config['title'] == "Parameters":
        value.update(info.data)  # Include any already-validated parameters as they might be needed
        config_settings = info.data.get("config_file_settings")
        value['config_file_settings'] = getattr(config_settings, info.field_name, {})
        value['tag'] = "Full"
        if value['algorithm'] is None and config_settings is not None:
            value['algorithm'] = config_settings.algorithm
    return value


class ParameterBase(BaseModel):
    tag: Literal['Base', 'Full']


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


def convert_to_array(value, dtype=None):
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
            return np.array(value, dtype=dtype)
        except ValueError:
            return value
    elif dtype is not None and value.dtype != dtype:
        return value.astype(dtype)
    return value


# Types
IntParameter = int | None
FloatParameter = float | None
ArrayParameter = Annotated[NDArray | None, BeforeValidator(convert_to_array)]
IntArrayParameter = Annotated[NDArray[np.int_] | None, BeforeValidator(partial(convert_to_array, dtype=np.int64))]
FloatArrayParameter = Annotated[NDArray[np.float64] | None, BeforeValidator(partial(convert_to_array, dtype=np.float64))]
FloatOrArrayParameter = Annotated[NDArray | float | None, BeforeValidator(partial(convert_to_array, dtype=np.float64))]
IntOrArrayParameter = Annotated[NDArray | int | None, BeforeValidator(partial(convert_to_array, dtype=np.int64))]


# Fields
AlwaysValidateNoneField = Field(None, validate_default=True)
ValidationModelField = Field(None, validate_default=True, json_schema_extra={'config_validation': True},
                             discriminator="tag")
