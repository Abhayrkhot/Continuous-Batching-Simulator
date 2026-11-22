"""batchsim: a discrete-event simulator for LLM serving schedulers."""

from batchsim.cost import (
    BatchStats,
    LinearStepModel,
    RooflineModel,
    StepTimeModel,
    load_cost_model,
)
from batchsim.engine import EngineConfig, SimulationError, Simulator, simulate
from batchsim.hardware import GPUS, MODELS, GPUSpec, ModelConfig, get_gpu, get_model
from batchsim.metrics import SLO, SimResult
from batchsim.request import Request, RequestSpec
from batchsim.schedulers import POLICIES, Scheduler, SchedulerConfig, make_scheduler
from batchsim.workload import Workload, azure, synthetic

__version__ = "0.2.0"

__all__ = [
    "GPUS",
    "MODELS",
    "POLICIES",
    "SLO",
    "BatchStats",
    "EngineConfig",
    "GPUSpec",
    "LinearStepModel",
    "ModelConfig",
    "Request",
    "RequestSpec",
    "RooflineModel",
    "Scheduler",
    "SchedulerConfig",
    "SimResult",
    "SimulationError",
    "Simulator",
    "StepTimeModel",
    "Workload",
    "azure",
    "get_gpu",
    "get_model",
    "load_cost_model",
    "make_scheduler",
    "simulate",
    "synthetic",
]
