from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.sparse import lil_matrix

if TYPE_CHECKING:
    from .base_sim_class import BaseSimClass


@dataclass
class CurrentData(ABC):
    """Data passed to each sim step"""

    @abstractmethod
    def update_population_array(self, population_array: lil_matrix, 
                                plot_idx: int) -> None:
        """Update the simulation population array with the current clone cell counts

        Parameters
        ----------
        population_array : lil_matrix
            The array storing the clone sizes at each sample point
        plot_idx : int
            The index of the column to update
        """
        ...

    @classmethod
    @abstractmethod
    def from_sim(cls, sim: "BaseSimClass") -> Self:
        """Create CurrentData object from simulation

        Parameters
        ----------
        sim : BaseSimClass
            A simulation instance

        Returns
        -------
        Self
            CurrentData, with the initial clone population
        """
        ...
    
    @abstractmethod
    def update(self, **kwargs)-> None:
        """Update the clone population

        Used during a simulation step when cells divide or mutate, or 
        labels are added
        """
        ... 


@dataclass
class NonSpatialCurrentData(CurrentData):
    """Tracks current cell populations for non-spatial algorithms

    To increase efficiency, we only list the cell counts for living 
    clones in the current population array. 
    So we also need to keep track of which clones those cells belong to. 
    """
    current_population: np.ndarray[tuple[int], np.dtype[np.int_]]  # Number of cells in each *living* clone
    non_zero_clones: np.ndarray[tuple[int], np.dtype[np.int_]]  # ids of clones with living cells

    @classmethod
    def from_sim(cls, sim: "BaseSimClass") -> Self:
        """Create NonSpatialCurrentData object from simulation

        Parameters
        ----------
        sim : BaseSimClass
            A non-spatial simulation

        Returns
        -------
        Self
            NonSpatialCurrentData with initial clone counts
        """
        current_population = np.zeros(len(sim.clones_array), dtype=int)
        current_population[:sim.initial_clones] = sim.initial_size_array

        non_zero_clones = np.where(current_population > 0)[0]
        current_population = current_population[non_zero_clones]

        return cls(current_population=current_population, non_zero_clones=non_zero_clones)    

    def update(self, current_population: np.ndarray[tuple[int], np.dtype[np.int_]], 
               non_zero_clones: np.ndarray[tuple[int], np.dtype[np.int_]]) -> None:
        """Update the current data

        Using a function to ensure all attributes are updated

        Parameters
        ----------
        current_population : np.ndarray[tuple[int], np.dtype[np.int_]]
             New cell count for each clone
        non_zero_clones : np.ndarray[tuple[int], np.dtype[np.int_]]
            New array of ids of surviving clones
        """
        self.current_population = current_population
        self.non_zero_clones = non_zero_clones

    def update_population_array(self, population_array: lil_matrix, 
                                plot_idx: int) -> None:
        """Update the simulation population array with the current 
        clone cell counts

        Parameters
        ----------
        population_array : lil_matrix
            The array storing the clone sizes at each sample point
        plot_idx : int
            The index of the column to update
        """
        population_array[self.non_zero_clones, plot_idx] = self.current_population
