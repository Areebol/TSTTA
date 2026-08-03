"""Online prototype memory used by TPA.

This module is intentionally independent from the CoBA runner.  It adds a
protected source bank, a replaceable online bank, a diverse delayed-label
anchor buffer, and replay-based prototype admission.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tta.tta_dual_utils.GCM import CoBA_TF_Adapter


@dataclass
class AnchorRecord:
    """A lightweight replay item; raw input sequences are never retained."""

    sample_id: int
    query: torch.Tensor
    y_base: torch.Tensor
    y_gt: torch.Tensor
    adapter_delta: torch.Tensor
    base_error: float

    def state_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "y_base": self.y_base,
            "y_gt": self.y_gt,
            "adapter_delta": self.adapter_delta,
            "base_error": self.base_error,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "AnchorRecord":
        return cls(
            sample_id=int(state["sample_id"]),
            query=state["query"].cpu(),
            y_base=state["y_base"].cpu(),
            y_gt=state["y_gt"].cpu(),
            adapter_delta=state["adapter_delta"].cpu(),
            base_error=float(state["base_error"]),
        )


class DiverseOnlineAnchorBuffer:
    """Keep diverse test-time anchors with nearest-pair competition.

    When the capacity is exceeded, the globally nearest pair in normalized
    query space competes.  The easier sample (smaller base-model error) is
    removed; an exact tie removes the older sample.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Anchor capacity must be positive.")
        self.capacity = int(capacity)
        self.records: List[AnchorRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def _nearest_pair(self) -> Tuple[int, int]:
        if len(self.records) < 2:
            raise RuntimeError("At least two records are required.")
        flat_queries = torch.stack(
            [record.query.float().reshape(-1) for record in self.records], dim=0
        )
        distances = torch.cdist(flat_queries, flat_queries, p=2)
        distances.fill_diagonal_(float("inf"))
        flat_index = int(torch.argmin(distances).item())
        n_records = distances.shape[0]
        first, second = divmod(flat_index, n_records)
        return (first, second) if first < second else (second, first)

    def add(self, record: AnchorRecord) -> Optional[int]:
        self.records.append(record)
        if len(self.records) <= self.capacity:
            return None

        first, second = self._nearest_pair()
        first_record = self.records[first]
        second_record = self.records[second]
        if first_record.base_error < second_record.base_error:
            remove_index = first
        elif second_record.base_error < first_record.base_error:
            remove_index = second
        else:
            remove_index = (
                first
                if first_record.sample_id < second_record.sample_id
                else second
            )
        return self.records.pop(remove_index).sample_id

    def state_dict(self) -> Dict:
        return {
            "capacity": self.capacity,
            "records": [record.state_dict() for record in self.records],
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"Anchor capacity mismatch: {state['capacity']} != {self.capacity}"
            )
        self.records = [
            AnchorRecord.from_state_dict(item) for item in state["records"]
        ]


class TPAPrototypeAdapter(CoBA_TF_Adapter):
    """CoBA frequency adapter plus protected/offline and online prototypes."""

    def __init__(
        self,
        *args,
        n_online: int = 16,
        target_start_idx: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if n_online <= 0:
            raise ValueError("The number of online prototype slots must be positive.")
        self.n_online = int(n_online)
        self.target_start_idx = int(target_start_idx)
        self.register_buffer(
            "online_keys",
            torch.zeros(self.n_var, self.n_online, self.feature_dim),
        )
        self.register_buffer(
            "online_values",
            torch.zeros(self.n_var, self.n_online, self.window_len),
        )
        self.register_buffer(
            "online_valid", torch.zeros(self.n_online, dtype=torch.bool)
        )
        self.register_buffer(
            "online_sample_ids",
            torch.full((self.n_online,), -1, dtype=torch.long),
        )

    @property
    def online_count(self) -> int:
        return int(self.online_valid.sum().item())

    def _select_target_channels(
        self, x: torch.Tensor, y_base: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[-1] == self.n_var:
            return x
        end_index = self.target_start_idx + self.n_var
        if end_index > x.shape[-1]:
            raise ValueError(
                "Cannot select forecast target channels from the input: "
                f"[{self.target_start_idx}:{end_index}] for {x.shape[-1]} variables."
            )
        selected = x[..., self.target_start_idx:end_index]
        if selected.shape[-1] != y_base.shape[-1]:
            raise ValueError(
                f"Input/output variable mismatch: {selected.shape[-1]} and "
                f"{y_base.shape[-1]}."
            )
        return selected

    def _get_query(self, x: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        selected_x = self._select_target_channels(x, y_base)
        combined = torch.cat([selected_x, y_base], dim=1)
        query = self.query_net(combined)
        return F.normalize(query, p=2, dim=-1)

    def make_bank(
        self,
        candidate_key: Optional[torch.Tensor] = None,
        candidate_value: Optional[torch.Tensor] = None,
        remove_online_slot: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a temporary bank without mutating persistent buffers."""

        keys: List[torch.Tensor] = [F.normalize(self.static_keys, p=2, dim=-1)]
        values: List[torch.Tensor] = [self.static_values]
        valid_slots = torch.nonzero(self.online_valid, as_tuple=False).flatten()
        if remove_online_slot is not None:
            valid_slots = valid_slots[valid_slots != int(remove_online_slot)]
        if valid_slots.numel() > 0:
            keys.append(self.online_keys[:, valid_slots])
            values.append(self.online_values[:, valid_slots])
        if (candidate_key is None) != (candidate_value is None):
            raise ValueError("Candidate key and value must be provided together.")
        if candidate_key is not None:
            keys.append(
                F.normalize(candidate_key.to(self.static_keys.device), p=2, dim=-1)
                .unsqueeze(1)
            )
            values.append(candidate_value.to(self.static_values.device).unsqueeze(1))
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def retrieve_from_query(
        self,
        query: torch.Tensor,
        bank_keys: Optional[torch.Tensor] = None,
        bank_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Retrieve a correction with shape [batch, horizon, variables]."""

        if bank_keys is None or bank_values is None:
            bank_keys, bank_values = self.make_bank()
        similarity = torch.einsum("bvd,vnd->bvn", query, bank_keys)
        weights = F.softmax(self.temperature * similarity, dim=-1)
        correction = torch.einsum("bvn,vnh->bvh", weights, bank_values)
        return correction.permute(0, 2, 1)

    def _frequency_delta(
        self, x: torch.Tensor, y_base: torch.Tensor
    ) -> torch.Tensor:
        selected_x = self._select_target_channels(x, y_base)
        combined = torch.cat([selected_x, y_base], dim=1)
        frequency = torch.fft.rfft(combined, dim=1, norm="ortho")
        if frequency.shape[1] != self.online_freq_r.shape[1]:
            raise ValueError(
                "Frequency length mismatch: "
                f"{frequency.shape[1]} != {self.online_freq_r.shape[1]}."
            )
        real = frequency.real
        imag = frequency.imag
        transformed_real = (
            real * self.online_freq_r
            - imag * self.online_freq_i
            + self.online_bias_r
        )
        transformed_imag = (
            real * self.online_freq_i
            + imag * self.online_freq_r
            + self.online_bias_i
        )
        transformed = torch.complex(transformed_real, transformed_imag)
        time_signal = torch.fft.irfft(
            transformed, n=combined.shape[1], dim=1, norm="ortho"
        )
        correction = time_signal[:, -self.window_len :, :]
        gate = torch.tanh(self.tafas_gating).view(1, 1, -1)
        return gate * correction

    def forward_with_details(
        self,
        y_base: torch.Tensor,
        x: torch.Tensor,
        bank_keys: Optional[torch.Tensor] = None,
        bank_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.online_mode:
            with torch.no_grad():
                query = self._get_query(x, y_base)
                bank_delta = self.retrieve_from_query(
                    query, bank_keys=bank_keys, bank_values=bank_values
                )
            frequency_delta = self._frequency_delta(x, y_base)
        else:
            query = self._get_query(x, y_base)
            bank_delta = self.retrieve_from_query(
                query, bank_keys=bank_keys, bank_values=bank_values
            )
            frequency_delta = torch.zeros_like(bank_delta)
        y_final = y_base + bank_delta + frequency_delta
        return y_final, {
            "query": query,
            "bank_delta": bank_delta,
            "frequency_delta": frequency_delta,
        }

    def forward(self, y_base: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(y_base, x)[0]

    @torch.no_grad()
    def commit_candidate(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        sample_id: int,
        replace_slot: Optional[int] = None,
    ) -> int:
        if replace_slot is None:
            free_slots = torch.nonzero(~self.online_valid, as_tuple=False).flatten()
            if free_slots.numel() == 0:
                raise RuntimeError("No free online prototype slot is available.")
            slot = int(free_slots[0].item())
        else:
            slot = int(replace_slot)
            if slot < 0 or slot >= self.n_online:
                raise IndexError(f"Invalid online prototype slot: {slot}")
        self.online_keys[:, slot].copy_(
            F.normalize(key.to(self.online_keys.device), p=2, dim=-1)
        )
        self.online_values[:, slot].copy_(value.to(self.online_values.device))
        self.online_valid[slot] = True
        self.online_sample_ids[slot] = int(sample_id)
        return slot


class TPAOnlineMemory:
    """Distill candidates and admit them only when replay MSE improves."""

    VALID_DISTILL_MODES = {"mean", "query_weighted"}

    def __init__(
        self,
        adapter: TPAPrototypeAdapter,
        anchor_capacity: int = 64,
        distill_mode: str = "mean",
        replay_batch_size: int = 128,
        coherence_eps: float = 1e-8,
    ):
        if distill_mode not in self.VALID_DISTILL_MODES:
            raise ValueError(
                f"Unknown distillation mode {distill_mode!r}; expected one of "
                f"{sorted(self.VALID_DISTILL_MODES)}."
            )
        self.adapter = adapter
        self.anchors = DiverseOnlineAnchorBuffer(anchor_capacity)
        self.distill_mode = distill_mode
        self.replay_batch_size = int(replay_batch_size)
        self.coherence_eps = float(coherence_eps)
        self.next_sample_id = 0
        self.update_history: List[Dict] = []

    @torch.no_grad()
    def distill_candidate(
        self,
        query: torch.Tensor,
        y_base: torch.Tensor,
        y_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return candidate key/value and a diagnostic coherence score."""

        query = query.detach()
        delta = (y_final - y_base).detach()
        candidate_key = F.normalize(query.mean(dim=0), p=2, dim=-1)
        if self.distill_mode == "mean":
            candidate_value = delta.mean(dim=0).permute(1, 0)
        else:
            similarity = torch.einsum("mvd,vd->mv", query, candidate_key)
            sample_weights = F.softmax(
                self.adapter.temperature * similarity, dim=0
            )
            candidate_value = torch.einsum(
                "mv,mhv->vh", sample_weights, delta
            )
        numerator = candidate_value.pow(2).sum()
        denominator = delta.pow(2).sum(dim=(1, 2)).mean()
        coherence = float(
            (numerator / denominator.clamp_min(self.coherence_eps)).item()
        )
        return candidate_key, candidate_value, coherence

    def _make_recent_records(
        self,
        query: torch.Tensor,
        y_base: torch.Tensor,
        y_gt: torch.Tensor,
        adapter_delta: torch.Tensor,
    ) -> List[AnchorRecord]:
        records = []
        for batch_index in range(query.shape[0]):
            sample_id = self.next_sample_id
            self.next_sample_id += 1
            base_error = float(
                F.mse_loss(y_base[batch_index], y_gt[batch_index]).item()
            )
            records.append(
                AnchorRecord(
                    sample_id=sample_id,
                    query=query[batch_index].detach().float().cpu(),
                    y_base=y_base[batch_index].detach().float().cpu(),
                    y_gt=y_gt[batch_index].detach().float().cpu(),
                    adapter_delta=adapter_delta[batch_index]
                    .detach()
                    .float()
                    .cpu(),
                    base_error=base_error,
                )
            )
        return records

    @staticmethod
    def _deduplicate(
        recent: Sequence[AnchorRecord], anchors: Iterable[AnchorRecord]
    ) -> List[AnchorRecord]:
        replay = list(recent)
        seen = {record.sample_id for record in recent}
        replay.extend(record for record in anchors if record.sample_id not in seen)
        return replay

    @torch.no_grad()
    def _replay_loss(
        self,
        records: Sequence[AnchorRecord],
        bank_keys: torch.Tensor,
        bank_values: torch.Tensor,
    ) -> float:
        if not records:
            return float("inf")
        device = self.adapter.static_keys.device
        squared_error = 0.0
        element_count = 0
        for start in range(0, len(records), self.replay_batch_size):
            batch = records[start : start + self.replay_batch_size]
            query = torch.stack([record.query for record in batch]).to(device)
            y_base = torch.stack([record.y_base for record in batch]).to(device)
            y_gt = torch.stack([record.y_gt for record in batch]).to(device)
            adapter_delta = torch.stack(
                [record.adapter_delta for record in batch]
            ).to(device)
            bank_delta = self.adapter.retrieve_from_query(
                query, bank_keys=bank_keys, bank_values=bank_values
            )
            prediction = y_base + bank_delta + adapter_delta
            squared_error += float(
                F.mse_loss(prediction, y_gt, reduction="sum").item()
            )
            element_count += y_gt.numel()
        return squared_error / element_count

    @torch.no_grad()
    def process_delayed_batch(
        self,
        query: torch.Tensor,
        y_base: torch.Tensor,
        y_final: torch.Tensor,
        y_gt: torch.Tensor,
        adapter_delta: torch.Tensor,
    ) -> Dict:
        """Propose one candidate after a fully labeled delayed batch."""

        candidate_key, candidate_value, coherence = self.distill_candidate(
            query, y_base, y_final
        )
        recent = self._make_recent_records(
            query, y_base, y_gt, adapter_delta
        )
        replay = self._deduplicate(recent, self.anchors.records)
        current_keys, current_values = self.adapter.make_bank()
        baseline_loss = self._replay_loss(replay, current_keys, current_values)

        accepted = False
        replacement_slot: Optional[int] = None
        best_loss = baseline_loss
        if self.adapter.online_count < self.adapter.n_online:
            expanded_keys, expanded_values = self.adapter.make_bank(
                candidate_key=candidate_key,
                candidate_value=candidate_value,
            )
            candidate_loss = self._replay_loss(
                replay, expanded_keys, expanded_values
            )
            if candidate_loss < baseline_loss:
                best_loss = candidate_loss
                accepted = True
        else:
            valid_slots = torch.nonzero(
                self.adapter.online_valid, as_tuple=False
            ).flatten()
            for slot_tensor in valid_slots:
                slot = int(slot_tensor.item())
                trial_keys, trial_values = self.adapter.make_bank(
                    candidate_key=candidate_key,
                    candidate_value=candidate_value,
                    remove_online_slot=slot,
                )
                trial_loss = self._replay_loss(replay, trial_keys, trial_values)
                if trial_loss < best_loss:
                    best_loss = trial_loss
                    replacement_slot = slot
                    accepted = True

        candidate_id = recent[-1].sample_id
        committed_slot = None
        if accepted:
            committed_slot = self.adapter.commit_candidate(
                candidate_key,
                candidate_value,
                sample_id=candidate_id,
                replace_slot=replacement_slot,
            )

        removed_anchor_ids = []
        for record in recent:
            removed_id = self.anchors.add(record)
            if removed_id is not None:
                removed_anchor_ids.append(removed_id)

        result = {
            "candidate_id": candidate_id,
            "distill_mode": self.distill_mode,
            "distill_coherence": coherence,
            "baseline_replay_mse": baseline_loss,
            "best_replay_mse": best_loss,
            "accepted": accepted,
            "committed_slot": committed_slot,
            "online_count": self.adapter.online_count,
            "replay_size": len(replay),
            "anchor_size": len(self.anchors),
            "removed_anchor_ids": removed_anchor_ids,
        }
        self.update_history.append(result)
        return result

    def state_dict(self) -> Dict:
        return {
            "distill_mode": self.distill_mode,
            "next_sample_id": self.next_sample_id,
            "anchors": self.anchors.state_dict(),
            "update_history": self.update_history,
        }

    def load_state_dict(self, state: Dict) -> None:
        if state["distill_mode"] != self.distill_mode:
            raise ValueError(
                f"Distillation mode mismatch: {state['distill_mode']} != "
                f"{self.distill_mode}"
            )
        self.next_sample_id = int(state["next_sample_id"])
        self.anchors.load_state_dict(state["anchors"])
        self.update_history = list(state["update_history"])
