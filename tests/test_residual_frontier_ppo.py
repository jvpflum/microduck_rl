import pytest
import torch
import torch.nn as nn

from mjlab_microduck.algorithms.residual_frontier_ppo import (
    _configure_surgical_output_training,
    _env_flag,
    _parse_trainable_action_rows,
    _remap_residual_as_frozen_base,
)


def test_parse_trainable_action_rows() -> None:
    assert _parse_trainable_action_rows(None, 14) is None
    assert _parse_trainable_action_rows("", 14) is None
    assert _parse_trainable_action_rows("10, 1, 10, 7", 14) == (1, 7, 10)


@pytest.mark.parametrize("value", ["nope", "-1", "14"])
def test_parse_trainable_action_rows_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_trainable_action_rows(value, 14)


def test_surgical_output_training_masks_features_and_unselected_rows() -> None:
    residual = nn.Sequential(nn.Linear(3, 5), nn.ELU(), nn.Linear(5, 4))
    _configure_surgical_output_training(residual, 4, (1, 3))

    residual(torch.ones(2, 3)).sum().backward()

    first = residual[0]
    output = residual[2]
    assert first.weight.grad is None
    assert first.bias.grad is None
    assert torch.count_nonzero(output.weight.grad[[0, 2]]) == 0
    assert torch.count_nonzero(output.bias.grad[[0, 2]]) == 0
    assert torch.count_nonzero(output.weight.grad[[1, 3]]) > 0
    assert torch.count_nonzero(output.bias.grad[[1, 3]]) > 0


def test_env_flag_parses_explicit_boolean_values(monkeypatch) -> None:
    monkeypatch.setenv("DUCKLAB_TEST_FLAG", "yes")
    assert _env_flag("DUCKLAB_TEST_FLAG") is True
    monkeypatch.setenv("DUCKLAB_TEST_FLAG", "off")
    assert _env_flag("DUCKLAB_TEST_FLAG") is False
    monkeypatch.setenv("DUCKLAB_TEST_FLAG", "maybe")
    with pytest.raises(ValueError):
        _env_flag("DUCKLAB_TEST_FLAG")


def test_remap_residual_as_frozen_base_preserves_donor_and_moves_residual() -> None:
    donor = torch.tensor([1.0])
    residual = torch.tensor([2.0])
    scale = torch.tensor(0.08)
    source = {
        "mlp.0.weight": donor,
        "residual_mlp.2.weight": residual,
        "_residual_max_action": scale,
    }

    remapped = _remap_residual_as_frozen_base(source)

    assert remapped["mlp.0.weight"] is donor
    assert remapped["frozen_residual_mlp.2.weight"] is residual
    assert remapped["_frozen_residual_max_action"] is scale
    assert "residual_mlp.2.weight" not in remapped
    assert "_residual_max_action" not in remapped
