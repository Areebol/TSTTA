"""Utilities for backpropagation-free online test-time adaptation.

The SPSA estimator follows the parameter-vector workflow used by ZOA, but uses
an antithetic (two-sided) finite difference to reduce estimator bias. All
fitness evaluations run under torch.no_grad(), so no activation graph is
retained for an online update.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn


LossClosure = Callable[[], Tensor]


def _iter_tensors(value: Any):
    """Yield tensors recursively from nested online-state containers."""
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def logical_tensor_bytes(*values: Any, cuda_only: bool = True) -> int:
    """Count logical tensor bytes once per tensor object.

    Logical bytes are appropriate for history-cache slices: a slice can share a
    much larger backing storage with the full test tensor, while a real streaming
    deployment only needs to retain the elements in the slice.
    """
    seen = set()
    total = 0
    for value in values:
        for tensor in _iter_tensors(value):
            if cuda_only and not tensor.is_cuda:
                continue
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            total += tensor.numel() * tensor.element_size()
    return total


def unique_storage_bytes(*values: Any, cuda_only: bool = True) -> int:
    """Count unique tensor storages, avoiding views and tied tensors twice."""
    seen = set()
    total = 0
    for value in values:
        for tensor in _iter_tensors(value):
            if cuda_only and not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr(), storage.nbytes())
            if key in seen:
                continue
            seen.add(key)
            total += storage.nbytes()
    return total


def optimizer_parameters(optimizer: torch.optim.Optimizer) -> List[nn.Parameter]:
    """Return optimizer parameters once, preserving parameter-group order."""
    parameters: List[nn.Parameter] = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in seen:
                parameters.append(parameter)
                seen.add(parameter_id)
    if not parameters:
        raise ValueError("The optimizer has no parameters to update.")
    return parameters


def parameters_to_vector(parameters: Iterable[nn.Parameter]) -> Tensor:
    """Flatten parameters into an independent vector without building a graph."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Cannot vectorize an empty parameter list.")
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters]).clone()


@torch.no_grad()
def vector_to_parameters(vector: Tensor, parameters: Iterable[nn.Parameter]) -> None:
    """Copy a flat vector into parameters without replacing Parameter storage."""
    pointer = 0
    for parameter in parameters:
        numel = parameter.numel()
        parameter.copy_(vector[pointer:pointer + numel].view_as(parameter))
        pointer += numel
    if pointer != vector.numel():
        raise ValueError(
            f"Parameter vector has {vector.numel()} elements, but {pointer} were consumed."
        )


@torch.no_grad()
def record_gradient(gradient: Tensor, parameters: Iterable[nn.Parameter]) -> None:
    """Write a flat estimated gradient to Parameter.grad for an optimizer."""
    pointer = 0
    for parameter in parameters:
        numel = parameter.numel()
        grad = gradient[pointer:pointer + numel].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = grad.clone()
        else:
            parameter.grad.copy_(grad)
        pointer += numel
    if pointer != gradient.numel():
        raise ValueError(
            f"Gradient vector has {gradient.numel()} elements, but {pointer} were consumed."
        )


def _sample_perturbation(vector: Tensor, distribution: str) -> Tensor:
    if distribution == "rademacher":
        return torch.empty_like(vector).bernoulli_(0.5).mul_(2).sub_(1)
    if distribution == "segmented_uniform":
        # ZOA distribution: U([-1, -0.5] union [0.5, 1]).
        magnitude = torch.empty_like(vector).uniform_(0.5, 1.0)
        sign = torch.empty_like(vector).bernoulli_(0.5).mul_(2).sub_(1)
        return magnitude.mul_(sign)
    raise ValueError(
        f"Unknown SPSA perturbation distribution {distribution!r}; expected "
        "'rademacher' or 'segmented_uniform'."
    )


@torch.no_grad()
def spsa_grad_estimate_bi(
    loss_closure: LossClosure,
    parameters: Sequence[nn.Parameter],
    perturbation_scale: float,
    sp_avg: int = 1,
    distribution: str = "rademacher",
) -> Tuple[Tensor, Tensor]:
    """Estimate a gradient with averaged, two-sided SPSA fitness queries.

    Returns (gradient, mean_center_loss) and always restores the unperturbed
    parameter vector, including when a fitness evaluation raises an exception.
    """
    if perturbation_scale <= 0:
        raise ValueError("perturbation_scale must be positive.")
    if sp_avg <= 0:
        raise ValueError("sp_avg must be positive.")

    center = parameters_to_vector(parameters)
    gradient = torch.zeros_like(center)
    loss_sum = torch.zeros((), device=center.device, dtype=center.dtype)

    try:
        for _ in range(sp_avg):
            perturbation = _sample_perturbation(center, distribution)

            vector_to_parameters(center + perturbation_scale * perturbation, parameters)
            loss_plus = loss_closure().detach()

            vector_to_parameters(center - perturbation_scale * perturbation, parameters)
            loss_minus = loss_closure().detach()

            directional_derivative = (
                (loss_plus - loss_minus) / (2.0 * perturbation_scale)
            )
            gradient.add_(directional_derivative / perturbation)
            loss_sum.add_((loss_plus + loss_minus) * 0.5)
    finally:
        vector_to_parameters(center, parameters)

    return gradient.div_(sp_avg), loss_sum.div_(sp_avg)


@torch.no_grad()
def spsa_grad_estimate_bi_blockwise(
    loss_closure: LossClosure,
    parameters: Sequence[nn.Parameter],
    perturbation_scale: float,
    sp_avg: int = 1,
    distribution: str = "rademacher",
) -> Tuple[Tensor, Tensor]:
    """Estimate each parameter tensor independently to reduce cross-block noise."""
    gradients = []
    losses = []
    for parameter in parameters:
        gradient, loss = spsa_grad_estimate_bi(
            loss_closure,
            [parameter],
            perturbation_scale=perturbation_scale,
            sp_avg=sp_avg,
            distribution=distribution,
        )
        gradients.append(gradient)
        losses.append(loss)
    return torch.cat(gradients), torch.stack(losses).mean()


def zero_order_step(
    optimizer: torch.optim.Optimizer,
    loss_closure: LossClosure,
    perturbation_scale: float,
    sp_avg: int = 1,
    distribution: str = "rademacher",
    blockwise: bool = False,
) -> Tensor:
    """Perform one optimizer step using a forward-only SPSA gradient."""
    parameters = optimizer_parameters(optimizer)
    optimizer.zero_grad(set_to_none=True)
    estimator = (
        spsa_grad_estimate_bi_blockwise if blockwise else spsa_grad_estimate_bi
    )
    gradient, loss = estimator(
        loss_closure=loss_closure,
        parameters=parameters,
        perturbation_scale=perturbation_scale,
        sp_avg=sp_avg,
        distribution=distribution,
    )
    record_gradient(gradient, parameters)
    optimizer.step()
    return loss


@dataclass
class OnlineMemoryProfiler:
    """Measure CUDA allocation peaks attributable to individual online steps."""

    enabled: bool = True
    num_updates: int = 0
    max_peak_allocated_bytes: int = 0
    max_peak_reserved_bytes: int = 0
    max_incremental_allocated_bytes: int = 0
    max_matched_no_grad_peak_bytes: int = 0
    max_matched_no_grad_incremental_bytes: int = 0
    max_update_excess_over_no_grad_bytes: int = 0
    max_saved_tensor_bytes: int = 0
    max_history_cache_logical_bytes: int = 0
    max_adapter_state_bytes: int = 0
    max_online_parameter_bytes: int = 0
    max_retrieval_projection_state_bytes: int = 0
    max_optimizer_state_bytes: int = 0
    max_gradient_state_bytes: int = 0
    max_adapter_loss_forward_incremental_bytes: int = 0
    total_update_seconds: float = 0.0
    max_update_seconds: float = 0.0
    _allocated_before: int = 0
    _matched_no_grad_incremental_bytes: int = 0
    _current_saved_tensor_bytes: int = 0
    _started_at: float = 0.0
    def measure_online_state(
        self,
        adapter: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Any,
        active_inputs: Any = (),
    ) -> None:
        """Measure persistent/logical state required only by online adaptation."""
        if not self.enabled:
            return

        adapter_tensors = list(adapter.parameters()) + list(adapter.buffers())
        online_parameters = optimizer_parameters(optimizer)
        online_storage_keys = set()
        for tensor in online_parameters:
            if tensor.is_cuda:
                storage = tensor.untyped_storage()
                online_storage_keys.add(
                    (tensor.device, storage.data_ptr(), storage.nbytes())
                )

        retrieval_bytes = 0
        seen = set()
        for tensor in adapter_tensors:
            if not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr(), storage.nbytes())
            if key in seen:
                continue
            seen.add(key)
            if key not in online_storage_keys:
                retrieval_bytes += storage.nbytes()

        optimizer_state = [
            value
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, Tensor)
        ]
        gradients = [
            parameter.grad
            for parameter in online_parameters
            if parameter.grad is not None
        ]

        self.max_history_cache_logical_bytes = max(
            self.max_history_cache_logical_bytes,
            logical_tensor_bytes(history, active_inputs),
        )
        self.max_adapter_state_bytes = max(
            self.max_adapter_state_bytes, unique_storage_bytes(adapter_tensors)
        )
        self.max_online_parameter_bytes = max(
            self.max_online_parameter_bytes,
            unique_storage_bytes(online_parameters),
        )
        self.max_retrieval_projection_state_bytes = max(
            self.max_retrieval_projection_state_bytes, retrieval_bytes
        )
        self.max_optimizer_state_bytes = max(
            self.max_optimizer_state_bytes, unique_storage_bytes(optimizer_state)
        )
        self.max_gradient_state_bytes = max(
            self.max_gradient_state_bytes, unique_storage_bytes(gradients)
        )

    def measure_adapter_loss_forward(self, loss_closure: LossClosure) -> None:
        """Measure adapter/query/loss forward workspace with base outputs live."""
        if not self.enabled or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with torch.no_grad():
                loss = loss_closure()
        del loss
        torch.cuda.synchronize()
        incremental = max(
            0, torch.cuda.max_memory_allocated() - allocated_before
        )
        self.max_adapter_loss_forward_incremental_bytes = max(
            self.max_adapter_loss_forward_incremental_bytes, incremental
        )


    def measure_matched_no_grad(self, loss_closure: LossClosure) -> None:
        """Measure the same adapter/loss forward without retaining autograd state."""
        if not self.enabled or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with torch.no_grad():
                baseline_loss = loss_closure()
        del baseline_loss
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        incremental = max(0, peak_allocated - allocated_before)
        self._matched_no_grad_incremental_bytes = incremental
        self.max_matched_no_grad_peak_bytes = max(
            self.max_matched_no_grad_peak_bytes, peak_allocated
        )
        self.max_matched_no_grad_incremental_bytes = max(
            self.max_matched_no_grad_incremental_bytes, incremental
        )

    def begin_step(self) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        self._allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self._started_at = time.perf_counter()

    def end_step(self) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - self._started_at
        self.total_update_seconds += update_seconds
        self.max_update_seconds = max(self.max_update_seconds, update_seconds)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        self.num_updates += 1
        self.max_peak_allocated_bytes = max(self.max_peak_allocated_bytes, peak_allocated)
        self.max_peak_reserved_bytes = max(self.max_peak_reserved_bytes, peak_reserved)
        update_incremental = max(0, peak_allocated - self._allocated_before)
        self.max_incremental_allocated_bytes = max(
            self.max_incremental_allocated_bytes, update_incremental
        )
        self.max_update_excess_over_no_grad_bytes = max(
            self.max_update_excess_over_no_grad_bytes,
            max(0, update_incremental - self._matched_no_grad_incremental_bytes),
        )

    @contextmanager
    def track_saved_tensors(self, excluded_tensors: Iterable[Tensor] = ()):
        """Track unique tensor storage retained for backward, excluding parameters."""
        if not self.enabled:
            yield
            return

        excluded_storages = set()
        for tensor in excluded_tensors:
            storage = tensor.untyped_storage()
            excluded_storages.add((storage.data_ptr(), storage.nbytes()))

        saved_storages = set()
        self._current_saved_tensor_bytes = 0

        def pack_hook(tensor):
            storage = tensor.untyped_storage()
            storage_key = (storage.data_ptr(), storage.nbytes())
            if (
                storage_key not in excluded_storages
                and storage_key not in saved_storages
            ):
                saved_storages.add(storage_key)
                self._current_saved_tensor_bytes += storage.nbytes()
            return tensor

        def unpack_hook(tensor):
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            yield
        self.max_saved_tensor_bytes = max(
            self.max_saved_tensor_bytes, self._current_saved_tensor_bytes
        )

    def summary(self) -> dict:
        mib = 1024.0 ** 2
        persistent_online_state = (
            self.max_adapter_state_bytes
            + self.max_history_cache_logical_bytes
            + self.max_optimizer_state_bytes
            + self.max_gradient_state_bytes
        )
        method_specific_update = max(
            self.max_saved_tensor_bytes,
            self.max_update_excess_over_no_grad_bytes,
        )
        accounted_lower_bound = persistent_online_state + max(
            self.max_adapter_loss_forward_incremental_bytes,
            method_specific_update,
        )
        accounted_upper_bound = (
            persistent_online_state
            + self.max_adapter_loss_forward_incremental_bytes
            + method_specific_update
        )
        return {
            "num_online_updates": self.num_updates,
            "peak_allocated_mb": self.max_peak_allocated_bytes / mib,
            "peak_reserved_mb": self.max_peak_reserved_bytes / mib,
            "peak_incremental_allocated_mb": self.max_incremental_allocated_bytes / mib,
            "peak_matched_no_grad_allocated_mb": (
                self.max_matched_no_grad_peak_bytes / mib
            ),
            "peak_matched_no_grad_incremental_mb": (
                self.max_matched_no_grad_incremental_bytes / mib
            ),
            "peak_update_excess_over_no_grad_mb": (
                self.max_update_excess_over_no_grad_bytes / mib
            ),
            "peak_saved_tensors_mb": self.max_saved_tensor_bytes / mib,
            "history_cache_logical_mb": (
                self.max_history_cache_logical_bytes / mib
            ),
            "adapter_state_mb": self.max_adapter_state_bytes / mib,
            "online_parameter_mb": self.max_online_parameter_bytes / mib,
            "retrieval_projection_state_mb": (
                self.max_retrieval_projection_state_bytes / mib
            ),
            "optimizer_state_mb": self.max_optimizer_state_bytes / mib,
            "gradient_state_mb": self.max_gradient_state_bytes / mib,
            "adapter_loss_forward_incremental_mb": (
                self.max_adapter_loss_forward_incremental_bytes / mib
            ),
            "persistent_online_state_mb": persistent_online_state / mib,
            "method_specific_update_mb": method_specific_update / mib,
            "non_forward_online_overhead_mb": (
                persistent_online_state + method_specific_update
            ) / mib,
            "accounted_online_adaptation_lower_bound_mb": accounted_lower_bound / mib,
            "accounted_online_adaptation_upper_bound_mb": accounted_upper_bound / mib,
            "accounted_online_adaptation_mb": (
                accounted_upper_bound
            ) / mib,
            "total_update_seconds": self.total_update_seconds,
            "mean_update_seconds": (
                self.total_update_seconds / self.num_updates if self.num_updates else 0.0
            ),
            "max_update_seconds": self.max_update_seconds,
        }
