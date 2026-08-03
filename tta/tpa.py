"""TPA runner: CoBA-TF adaptation with delayed online prototype management."""

import json
import os

import torch

from models.forecast import forecast
from utils.misc import prepare_inputs
from tta.coba import Adapter as CoBAAdapter
from tta.loss import CoBA_Loss
from tta.tpa_memory import TPAOnlineMemory, TPAPrototypeAdapter


class Adapter(CoBAAdapter):
    """Add prototype admission after each fully labeled delayed batch."""

    def __init__(self, cfg, model, norm_module=None):
        if cfg.TTA.DUAL.CALI_NAME != "TPAPrototypeAdapter":
            raise ValueError(
                "TPA requires TTA.DUAL.CALI_NAME=TPAPrototypeAdapter."
            )
        if cfg.TTA.DUAL.CALI_INPUT_ENABLE:
            raise ValueError("TPA currently requires CALI_INPUT_ENABLE=False.")
        if not cfg.TTA.DUAL.CALI_OUTPUT_ENABLE:
            raise ValueError("TPA requires CALI_OUTPUT_ENABLE=True.")
        if not cfg.TTA.DUAL.COBA_ONLINE_ENABLED:
            raise ValueError("TPA requires COBA_ONLINE_ENABLED=True.")
        super().__init__(cfg, model, norm_module=norm_module)
        if not isinstance(self.cali.out_cali, TPAPrototypeAdapter):
            raise TypeError("The constructed output adapter is not a TPA adapter.")
        # Source knowledge is immutable after offline pretraining. The online
        # optimizer already contains only frequency-adapter parameters; these
        # flags also prevent accidental source gradients from accumulating.
        self.cali.out_cali.query_net.requires_grad_(False)
        self.cali.out_cali.static_keys.requires_grad_(False)
        self.cali.out_cali.static_values.requires_grad_(False)
        self.tpa_memory = TPAOnlineMemory(
            adapter=self.cali.out_cali,
            anchor_capacity=cfg.TTA.TPA.ANCHOR_CAPACITY,
            distill_mode=cfg.TTA.TPA.DISTILL_MODE,
            replay_batch_size=cfg.TTA.TPA.REPLAY_BATCH_SIZE,
            coherence_eps=cfg.TTA.TPA.COHERENCE_EPS,
        )
        self.save_name = (
            f"tpa-{cfg.TTA.TPA.DISTILL_MODE}"
            f"-src-{cfg.TTA.TPA.N_SOURCE:02d}"
            f"-online-{cfg.TTA.TPA.N_ONLINE:02d}"
            f"-lr-{cfg.TTA.DUAL.COBA_ONLINE_LR:.5f}"
        )

    def _online_loss(self, prediction, ground_truth):
        if isinstance(self.loss_fn, CoBA_Loss):
            return self.loss_fn(
                prediction,
                ground_truth,
                bases=self.cali.out_cali.static_keys,
            )
        return self.loss_fn(prediction, ground_truth)

    def _adapt_with_full_ground_truth_if_available(self):
        while (
            self.pred_step_end_dict
            and self.cur_step
            >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]
        ):
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)
            enc_history = prepare_inputs(inputs_history)[0]

            for _ in range(self.cfg.TTA.DUAL.STEPS):
                self.n_adapt += 1
                self._switch_model_to_train()
                with torch.no_grad():
                    y_base, ground_truth = forecast(
                        self.cfg,
                        inputs_history,
                        self.model,
                        self.norm_module,
                    )
                y_final = self.cali.output_calibration(y_base, enc_history)
                loss = self._online_loss(y_final, ground_truth)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self._switch_model_to_eval()
            with torch.no_grad():
                y_base, ground_truth = forecast(
                    self.cfg,
                    inputs_history,
                    self.model,
                    self.norm_module,
                )
                y_final, details = self.cali.out_cali.forward_with_details(
                    y_base, enc_history
                )
                update = self.tpa_memory.process_delayed_batch(
                    query=details["query"],
                    y_base=y_base,
                    y_final=y_final,
                    y_gt=ground_truth,
                    adapter_delta=details["frequency_delta"],
                )
            print(
                "[TPA] candidate={candidate_id} accepted={accepted} "
                "replay={baseline_replay_mse:.6f}->{best_replay_mse:.6f} "
                "online={online_count} coherence={distill_coherence:.4f}".format(
                    **update
                )
            )
            self.pred_step_end_dict.pop(batch_idx_available)

    def _report(self):
        super()._report()
        os.makedirs(self.cfg.RESULT_DIR, exist_ok=True)
        history_path = os.path.join(
            self.cfg.RESULT_DIR, f"{self.save_name}-prototype-updates.json"
        )
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(
                self.tpa_memory.update_history,
                file,
                ensure_ascii=False,
                indent=2,
            )
        if self.cfg.TTA.TPA.SAVE_STATE:
            state_path = os.path.join(
                self.cfg.RESULT_DIR, f"{self.save_name}-prototype-state.pt"
            )
            torch.save(
                {
                    "adapter": self.cali.out_cali.state_dict(),
                    "memory": self.tpa_memory.state_dict(),
                },
                state_path,
            )


def build_adapter(cfg, model, norm_module=None):
    return Adapter(cfg, model, norm_module=norm_module)
