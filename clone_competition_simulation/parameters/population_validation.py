from typing import Annotated, Literal
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Tag,
    BeforeValidator
)
from .validation_utils import (
    assign_config_settings,
    IntParameter,
    ArrayParameter,
    IntArrayParameter,
    AlwaysValidateNoneField,
    ValidationBase
)
from loguru import logger


class PopulationParameters(BaseModel):
    tag: Literal['Base'] = 'Base'
    model_config = ConfigDict(arbitrary_types_allowed=True)
    initial_cells: IntParameter = AlwaysValidateNoneField
    initial_size_array: IntArrayParameter = AlwaysValidateNoneField
    grid_shape: tuple[int, int] | None = AlwaysValidateNoneField
    initial_grid: IntArrayParameter = AlwaysValidateNoneField
    population_limit: int | None = AlwaysValidateNoneField
    cell_in_own_neighbourhood: bool | None = AlwaysValidateNoneField


class PopulationValidator(PopulationParameters, ValidationBase):
    tag: Literal['Full']
    config_file_settings: PopulationParameters | None = None

    def _validate_model(self):
        """Checks that only one population parameter has been given"""
        self.initial_cells = self.get_value_from_config("initial_cells")
        self.initial_size_array = self.get_value_from_config("initial_size_array")
        self.grid_shape = self.get_value_from_config("grid_shape")
        self.initial_grid = self.get_value_from_config("initial_grid")
        num_defined = sum([self.initial_cells is not None, self.initial_size_array is not None,
                           self.grid_shape is not None, self.initial_grid is not None])
        if num_defined == 0:
            raise ValueError('Must provide one of:\n\tinitial_cells\n\tinitial_size_array\n\t'
                             'grid_shape (Moran2D/WF2D only)\n\tinitial_grid (Moran2D/WF2D only)')
        if self.algorithm.two_dimensional:
            self._setup_2D_initial_population()
        else:
            if self.initial_cells is None and self.initial_size_array is None:
                raise ValueError('Must provide initial_cells or initial_size_array')
            self._setup_initial_population_non_spatial()

    def _setup_2D_initial_population(self):
        if self.initial_cells is not None:
            self._try_making_square_grid()
            self.initial_size_array = np.array([self.initial_cells], dtype=int)
            self.initial_grid = np.zeros(self.grid_shape, dtype=int)
        elif self.initial_size_array is not None:
            if len(self.initial_size_array) == 1:
                self.initial_cells = sum(self.initial_size_array)
                self._try_making_square_grid()
                self.initial_grid = np.zeros(self.grid_shape, dtype=int)
            else:
                raise ValueError('Cannot use initial_size_array with 2D simulation. Provide initial_grid instead.')
        elif self.grid_shape is not None:
            self.initial_cells = self.grid_shape[0] * self.grid_shape[1]
            self.initial_size_array = np.array([self.initial_cells], dtype=int)
            self.initial_grid = np.zeros(self.grid_shape, dtype=int)
        elif self.initial_grid is not None:
            self.grid_shape = self.initial_grid.shape
            self.initial_cells = self.grid_shape[0] * self.grid_shape[1]
            self.initial_grid = self.initial_grid.astype(int)
            self._create_initial_size_array_from_grid()
        else:
            raise ValueError('Please provide one of the population size inputs')

        # Check that the hexagonal grid has only even dimensions.
        if self.grid_shape[0] % 2 != 0 or self.grid_shape[1] % 2 != 0:
            raise ValueError('Must have even number of rows/columns in the hexagonal grid.')

        self.cell_in_own_neighbourhood = self.get_value_from_config("cell_in_own_neighbourhood")
        if self.cell_in_own_neighbourhood is None:
            raise ValueError("Must provide cell_in_own_neighbourhood to run a 2D simulation")

    def _try_making_square_grid(self):
        poss_grid_size = int(np.sqrt(self.initial_cells))
        if poss_grid_size ** 2 == self.initial_cells:
            self.grid_shape = (poss_grid_size, poss_grid_size)
            logger.debug(f'Using a grid of {self.grid_shape[0]}x{self.grid_shape[1]}')
        else:
            raise ValueError(f'Square grid not compatible with {self.initial_cells} cells. '
                                  f'To run a rectangular grid provide a grid shape or initial grid')

    def _create_initial_size_array_from_grid(self):
        """For the 2D simulations, if an initial grid of clone positions is provided, fill in the initial_size_array"""

        # If the initial size array is not given, define it here.
        idx_counts = {k:v for k,v in zip(*np.unique(self.initial_grid, return_counts=True))}
        self.initial_size_array = []
        for i in range(int(max(idx_counts))+1):
            if i in idx_counts:
                self.initial_size_array.append(idx_counts[i])
            else:
                self.initial_size_array.append(0)
        self.initial_size_array = np.array(self.initial_size_array, dtype=int)

    def _setup_initial_population_non_spatial(self):
        if self.initial_cells is not None:
            self.initial_size_array = np.array([self.initial_cells], dtype=int)
        elif self.initial_size_array is not None:
            self.initial_cells = sum(self.initial_size_array)


population_validation_type = Annotated[
    (Annotated[PopulationParameters, Tag("Base")] | Annotated[PopulationValidator, Tag("Full")]),
    BeforeValidator(assign_config_settings)
]
