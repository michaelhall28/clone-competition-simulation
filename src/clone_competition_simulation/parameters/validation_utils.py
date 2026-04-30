from functools import partial
from typing import Annotated, Any, Self, Literal, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator, BeforeValidator, ConfigDict

from .algorithm_validation import Algorithm


class ParameterBase(BaseModel):
    _field_name: ClassVar[str]
    tag: Literal['Base', 'Full']

    # Be strict at this stage so we catch typos in the input. 
    model_config = ConfigDict(extra='forbid')


class ValidationBase(BaseModel):
    algorithm: Algorithm
    validated: bool = False
    config_file_settings: ParameterBase | None = None

    # Be less strict here so we ignore arguments meant for other settings. 
    model_config = ConfigDict(extra='ignore')

    def _validate_model(self) -> None:
        raise NotImplementedError

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
    

def assign_config_settings(value, info) -> dict[str, Any]:
    if value is not None:
        if not isinstance(value, ParameterBase):
            # Run through the basic pydantic input validation if not done already
            value = find_subclass(ParameterBase, info.field_name)(**value)
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


def find_subclass(cls: type[ParameterBase], field_name: str) -> type[ParameterBase]:
    """
    Find the right ParameterBase subclass for the parameter field
    """
    for subcls in cls.__subclasses__():
        if subcls._field_name == field_name:
            return subcls
    raise ValueError(f"No ParameterClass for field name {field_name}")


def convert_to_array(value, dtype=None) -> Any | int | float | NDArray[np.float64] | NDArray[np.int_]:
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
