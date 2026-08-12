import pytest
import torch

from tta.zo_utils import (
    OnlineMemoryProfiler,
    logical_tensor_bytes,
    parameters_to_vector,
    record_gradient,
    spsa_grad_estimate_bi,
    vector_to_parameters,
    zero_order_step,
)


def test_parameter_vector_round_trip_and_gradient_recording():
    first = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
    second = torch.nn.Parameter(torch.tensor([-3.0]))
    parameters = [first, second]

    vector = parameters_to_vector(parameters)
    assert torch.equal(vector, torch.tensor([1.0, 2.0, -3.0]))

    vector_to_parameters(torch.tensor([4.0, 5.0, 6.0]), parameters)
    assert torch.equal(first, torch.tensor([[4.0, 5.0]]))
    assert torch.equal(second, torch.tensor([6.0]))

    record_gradient(torch.tensor([0.1, 0.2, 0.3]), parameters)
    assert torch.allclose(first.grad, torch.tensor([[0.1, 0.2]]))
    assert torch.allclose(second.grad, torch.tensor([0.3]))


@pytest.mark.parametrize("distribution", ["rademacher", "segmented_uniform"])
def test_two_sided_spsa_is_exact_for_one_dimensional_quadratic(distribution):
    parameter = torch.nn.Parameter(torch.tensor([2.0]))

    def closure():
        return (parameter ** 2).sum()

    gradient, center_loss = spsa_grad_estimate_bi(
        closure,
        [parameter],
        perturbation_scale=0.1,
        sp_avg=3,
        distribution=distribution,
    )

    assert torch.allclose(gradient, torch.tensor([4.0]), atol=1e-5)
    assert center_loss.item() > closure().item()
    assert torch.equal(parameter, torch.tensor([2.0]))


def test_zero_order_step_updates_without_autograd_graph():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def closure():
        assert not torch.is_grad_enabled()
        return (parameter ** 2).sum()

    zero_order_step(optimizer, closure, perturbation_scale=0.01)
    assert torch.allclose(parameter, torch.tensor([1.6]), atol=1e-4)


def test_saved_tensor_profiler_sees_bp_graph_but_not_no_grad_forward():
    profiler = OnlineMemoryProfiler()
    parameter = torch.nn.Parameter(torch.ones(32))
    inputs = torch.ones(32)

    with profiler.track_saved_tensors([parameter]):
        (parameter * inputs).square().sum().backward()
    assert profiler.summary()["peak_saved_tensors_mb"] > 0
    assert (
        profiler._current_saved_tensor_bytes
        >= inputs.untyped_storage().nbytes()
    )

    with profiler.track_saved_tensors([parameter]), torch.no_grad():
        (parameter * inputs).square().sum()
    assert profiler._current_saved_tensor_bytes == 0


def test_memory_summary_exposes_matched_forward_subtraction_fields():
    summary = OnlineMemoryProfiler(enabled=False).summary()
    assert summary["peak_matched_no_grad_allocated_mb"] == 0
    assert summary["peak_matched_no_grad_incremental_mb"] == 0
    assert summary["peak_update_excess_over_no_grad_mb"] == 0
    assert summary["accounted_online_adaptation_mb"] == 0


def test_logical_tensor_bytes_deduplicates_objects_but_counts_views_logically():
    tensor = torch.ones(8)
    view = tensor[:4]
    assert logical_tensor_bytes(tensor, tensor, cuda_only=False) == 8 * 4
    assert logical_tensor_bytes(view, cuda_only=False) == 4 * 4



def test_blockwise_step_updates_independent_scalar_blocks_exactly():
    first = torch.nn.Parameter(torch.tensor([2.0]))
    second = torch.nn.Parameter(torch.tensor([-3.0]))
    optimizer = torch.optim.SGD([first, second], lr=0.1)

    def closure():
        return first.square().sum() + second.square().sum()

    zero_order_step(
        optimizer,
        closure,
        perturbation_scale=0.01,
        blockwise=True,
    )
    assert torch.allclose(first, torch.tensor([1.6]), atol=1e-4)
    assert torch.allclose(second, torch.tensor([-2.4]), atol=1e-4)
