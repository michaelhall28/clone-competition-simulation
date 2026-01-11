from src.clone_competition_simulation.parameters import Parameters
from src.clone_competition_simulation.parameters.times_validation import TimeValidator, TimeParameters
from src.clone_competition_simulation.parameters.population_validation import PopulationValidator, PopulationParameters
from src.clone_competition_simulation.parameters.fitness_validation import FitnessValidator, FitnessParameters
from src.clone_competition_simulation.parameters.label_validation import LabelValidator, LabelParameters
from src.clone_competition_simulation.parameters.treatment_validation import TreatmentValidator, TreatmentParameters
from src.clone_competition_simulation.parameters.plotting_validation import PlottingValidator, PlottingParameters
from src.clone_competition_simulation.parameters.differentiated_cells_validation import (
    DifferentiatedCellsValidator,
    DifferentiatedCellsParameters
)


def test_parameter_object_discrimination():
    """
    The parameters are split into sections (times, population, etc).
    There is a Base class for each section which is used to input any parameters and does not do any complex validation.
    The "Full" class does the whole validation.

    We need to make sure that the validated classes are used for validation, and the base classes are used for input.

    Returns:

    """
    sim_parameters = Parameters(
        algorithm="WF",
        population=PopulationParameters(initial_cells=100),
        times=TimeParameters(max_time=10, division_rate=1)
    )

    assert isinstance(sim_parameters.population, PopulationValidator)
    assert isinstance(sim_parameters.config_file_settings.population, PopulationParameters)
    assert not isinstance(sim_parameters.config_file_settings.population, PopulationValidator)

    assert isinstance(sim_parameters.times, TimeValidator)
    assert isinstance(sim_parameters.config_file_settings.times, TimeParameters)
    assert not isinstance(sim_parameters.config_file_settings.times, TimeValidator)
    assert isinstance(sim_parameters.times.population, PopulationValidator)

    assert isinstance(sim_parameters.fitness, FitnessValidator)
    assert isinstance(sim_parameters.config_file_settings.fitness, FitnessParameters)
    assert not isinstance(sim_parameters.config_file_settings.fitness, FitnessValidator)
    assert isinstance(sim_parameters.fitness.population, PopulationValidator)
    assert isinstance(sim_parameters.fitness.times, TimeValidator)

    assert isinstance(sim_parameters.differentiated_cells, DifferentiatedCellsValidator)
    assert isinstance(sim_parameters.config_file_settings.differentiated_cells, DifferentiatedCellsParameters)
    assert not isinstance(sim_parameters.config_file_settings.differentiated_cells, DifferentiatedCellsValidator)
    assert isinstance(sim_parameters.differentiated_cells.times, TimeValidator)

    assert isinstance(sim_parameters.labels, LabelValidator)
    assert isinstance(sim_parameters.config_file_settings.labels, LabelParameters)
    assert not isinstance(sim_parameters.config_file_settings.labels, LabelValidator)
    assert isinstance(sim_parameters.labels.population, PopulationValidator)
    assert isinstance(sim_parameters.labels.fitness, FitnessValidator)

    assert isinstance(sim_parameters.treatment, TreatmentValidator)
    assert isinstance(sim_parameters.config_file_settings.treatment, TreatmentParameters)
    assert not isinstance(sim_parameters.config_file_settings.treatment, TreatmentValidator)
    assert isinstance(sim_parameters.treatment.population, PopulationValidator)
    assert isinstance(sim_parameters.treatment.fitness, FitnessValidator)

    assert isinstance(sim_parameters.plotting, PlottingValidator)
    assert isinstance(sim_parameters.config_file_settings.plotting, PlottingParameters)
    assert not isinstance(sim_parameters.config_file_settings.plotting, PlottingValidator)





