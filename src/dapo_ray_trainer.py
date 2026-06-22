# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import glob as glob_module
import json
import os
import shutil
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


def _json_default(obj):
    """JSON serializer for numpy scalars and other non-serializable types."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _init_trajectory_log_dir(self, resumed_from_step: int = 0):
        """
        Initialize the trajectory log directory under <cwd>/log_data/<project>/<exp>/.

        - resumed_from_step == 0: fresh start, wipe the entire directory if it exists.
        - resumed_from_step > 0:  recovering from a checkpoint at that step, keep
          files for steps <= resumed_from_step and delete everything after it.

        Sub-directories:
          train/  -> one file per training step: step_{N:06d}.jsonl
          val/    -> one file per validation call: step_{N:06d}.jsonl
        """
        project_name = self.config.trainer.project_name
        exp_name = self.config.trainer.experiment_name
        base_dir = os.path.join(os.getcwd(), "log_data", project_name, exp_name)

        if resumed_from_step == 0:
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
            print(f"[TrajectoryLog] Initialized fresh log directory: {base_dir}")
        else:
            os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
            for sub in ("train", "val"):
                pattern = os.path.join(base_dir, sub, "step_*.jsonl")
                for fpath in glob_module.glob(pattern):
                    fname = os.path.basename(fpath)
                    try:
                        step_num = int(fname.replace("step_", "").replace(".jsonl", ""))
                    except ValueError:
                        continue
                    if step_num > resumed_from_step:
                        os.remove(fpath)
                        print(f"[TrajectoryLog] Removed stale log (step {step_num} > {resumed_from_step}): {fpath}")
            print(f"[TrajectoryLog] Resumed log directory at step {resumed_from_step}: {base_dir}")

        self._traj_log_base_dir = base_dir

    def _save_trajectories(self, split: str, step: int, inputs: list, outputs: list, scores: list,
                           extra: dict = None):
        """
        Save trajectory records to <base_dir>/<split>/step_{step:06d}.jsonl.
        Each line is a JSON object with keys: input, output, score, [extra fields].
        """
        out_path = os.path.join(self._traj_log_base_dir, split, f"step_{step:06d}.jsonl")
        extra = extra or {}
        with open(out_path, "w", encoding="utf-8") as f:
            for i, (inp, out, sc) in enumerate(zip(inputs, outputs, scores)):
                record = {"step": step, "idx": i, "input": inp, "output": out, "score": sc}
                for k, v in extra.items():
                    vals = v if isinstance(v, (list, np.ndarray)) else []
                    record[k] = vals[i] if i < len(vals) else None
                f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        print(f"[TrajectoryLog] Saved {len(inputs)} trajectories → {out_path}")

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        os.makedirs(local_global_step_folder, exist_ok=True)
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, "
                "set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic"
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # ---- Clean up old checkpoints: keep only HF weights ----
        for ckpt_dir in glob_module.glob(
            os.path.join(self.config.trainer.default_local_dir, "global_step_*")
        ):
            if os.path.basename(ckpt_dir) == f"global_step_{self.global_steps}":
                continue  # skip the checkpoint we just saved

            # Remove dataloader state
            data_pt = os.path.join(ckpt_dir, "data.pt")
            if os.path.exists(data_pt):
                try:
                    os.remove(data_pt)
                    print(f"Removed old dataloader state: {data_pt}")
                except Exception as e:
                    print(f"Failed to remove {data_pt}: {e}")

            for role in ["actor", "critic"]:
                role_dir = os.path.join(ckpt_dir, role)
                if not os.path.isdir(role_dir):
                    continue
                hf_dir = os.path.join(role_dir, "huggingface")
                if not os.path.isdir(hf_dir):
                    print(f"Warning: {role_dir} has no huggingface/ subdir, skipping cleanup")
                    continue
                # Delete everything inside role_dir except huggingface/
                for item in os.listdir(role_dir):
                    if item == "huggingface":
                        continue
                    item_path = os.path.join(role_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        print(f"Removed old non-HF file: {item_path}")
                    except Exception as e:
                        print(f"Failed to remove {item_path}: {e}")

    def _record_and_prune_checkpoints(self, val_metrics: dict, keep_best: int = 3):
        """Keep only the best `keep_best` checkpoints (ranked by AIME pass@1) plus the
        current step's checkpoint; delete all others entirely.

        Called after validation so the current step's pass@1 is available. The metric
        history is persisted to `checkpoint_metrics.json` under default_local_dir so
        rankings survive a resume.
        """
        ckpt_root = self.config.trainer.default_local_dir
        cur_step = self.global_steps
        cur_dir = os.path.join(ckpt_root, f"global_step_{cur_step}")
        if not os.path.isdir(cur_dir):
            return  # no checkpoint saved this step, nothing to prune

        # Extract AIME pass@1 from val metrics (data_source containing "aime").
        aime_score = None
        for k, v in val_metrics.items():
            kl = k.lower()
            if kl.startswith("val-core/") and "aime" in kl and kl.endswith("/pass_k/pass@1"):
                aime_score = float(v)
                break
        if aime_score is None:
            print("[CkptPrune] No AIME pass@1 found in val metrics; skipping prune this step.")
            return

        hist_path = os.path.join(ckpt_root, "checkpoint_metrics.json")
        history = {}
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    history = json.load(f)
            except Exception:
                history = {}
        history[str(cur_step)] = aime_score

        # Collect existing checkpoint steps on disk.
        existing_steps = []
        for d in glob_module.glob(os.path.join(ckpt_root, "global_step_*")):
            if not os.path.isdir(d):
                continue
            try:
                existing_steps.append(int(os.path.basename(d).replace("global_step_", "")))
            except ValueError:
                continue

        # Keep the top `keep_best` by AIME pass@1, plus the current step.
        ranked = sorted(existing_steps, key=lambda s: history.get(str(s), float("-inf")), reverse=True)
        keep = set(ranked[:keep_best])
        keep.add(cur_step)

        for s in existing_steps:
            if s in keep:
                continue
            d = os.path.join(ckpt_root, f"global_step_{s}")
            try:
                shutil.rmtree(d)
                print(f"[CkptPrune] Removed checkpoint not in best-{keep_best}+current: {d}")
            except Exception as e:
                print(f"[CkptPrune] Failed to remove {d}: {e}")

        history = {s: sc for s, sc in history.items() if int(s) in keep}
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(
            f"[CkptPrune] Kept {sorted(keep)} "
            f"(best-{keep_best} by AIME pass@1 + current step {cur_step}); "
            f"current pass@1={aime_score:.4f}"
        )

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            old_log_prob_metrics = {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_response_lengths = []

        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            # Decode outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Response lengths (non-padding tokens)
            response_len = output_ids.shape[1]
            if "attention_mask" in test_output_gen_batch.batch:
                resp_attention_mask = test_output_gen_batch.batch["attention_mask"][:, -response_len:]
            else:
                pad_token_id = self.tokenizer.pad_token_id
                resp_attention_mask = (output_ids != pad_token_id).long()
            actual_lengths = resp_attention_mask.sum(dim=-1).cpu().tolist()
            sample_response_lengths.extend(actual_lengths)

            test_batch = test_batch.union(test_output_gen_batch)

            # Decode inputs (prompts available after generation)
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, lst in reward_extra_info.items():
                reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        # Save validation trajectories to disk
        if hasattr(self, "_traj_log_base_dir"):
            extra_info = {k: v for k, v in reward_extra_infos_dict.items() if k != "reward"}
            self._save_trajectories(
                split="val",
                step=self.global_steps,
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                extra=extra_info if extra_info else None,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), \
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = {}

        # pass@k and length metrics per data source
        score_key = "acc" if "acc" in reward_extra_infos_dict else "reward"
        score_vals = reward_extra_infos_dict[score_key]

        ds_prompt2scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        ds_prompt2lengths: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for idx, (ds, prompt) in enumerate(zip(data_sources, sample_inputs)):
            ds_prompt2scores[ds][prompt].append(score_vals[idx])
            ds_prompt2lengths[ds][prompt].append(sample_response_lengths[idx])

        def _pass_at_k(n: int, c: int, k: int) -> float:
            """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
            if n - c < k:
                return 1.0
            from math import comb
            return 1.0 - comb(n - c, k) / comb(n, k)

        for ds in ds_prompt2scores:
            prompt2scores = ds_prompt2scores[ds]
            prompt2lengths = ds_prompt2lengths[ds]

            n_values = [len(v) for v in prompt2scores.values()]
            n = n_values[0] if n_values else 1

            per_question_mean_acc = []
            per_question_acc_var = []
            pass_k_accum: dict[int, list] = defaultdict(list)

            ks = []
            k = 1
            while k <= n:
                ks.append(k)
                k *= 2

            for prompt, scores in prompt2scores.items():
                n_q = len(scores)
                correct = [1 if s > 0 else 0 for s in scores]
                c = sum(correct)
                p = float(np.mean(correct))
                per_question_mean_acc.append(p)
                # within-question variance: p*(1-p), meaningful only when n > 1
                per_question_acc_var.append(p * (1.0 - p))
                for ki in ks:
                    if ki <= n_q:
                        pass_k_accum[ki].append(_pass_at_k(n_q, c, ki))

            all_lengths = [l for lengths in prompt2lengths.values() for l in lengths]
            length_mean = float(np.mean(all_lengths)) if all_lengths else 0.0
            mean_acc = float(np.mean(per_question_mean_acc)) if per_question_mean_acc else 0.0
            # within-question acc variance averaged over all questions
            acc_var_within = float(np.mean(per_question_acc_var)) if per_question_acc_var else 0.0
            # cross-question acc variance: spread of difficulty across questions
            acc_var_across = float(np.var(per_question_mean_acc)) if len(per_question_mean_acc) > 1 else 0.0

            print(f"[val] data_source={ds}, n={n}, mean_acc={mean_acc:.4f}, "
                  f"acc_var_within={acc_var_within:.4f}, acc_var_across={acc_var_across:.4f}, "
                  f"length_mean={length_mean:.1f}")
            for ki in ks:
                pass_k_val = float(np.mean(pass_k_accum[ki])) if pass_k_accum[ki] else 0.0
                print(f"  pass@{ki}={pass_k_val:.4f}")
                metric_dict[f"val-core/{ds}/pass_k/pass@{ki}"] = pass_k_val
            metric_dict[f"val-core/{ds}/mean_acc"] = mean_acc
            metric_dict[f"val-core/{ds}/acc_var/within_question"] = acc_var_within
            metric_dict[f"val-core/{ds}/acc_var/across_question"] = acc_var_across
            metric_dict[f"val-core/{ds}/response_length/mean"] = length_mean

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # Initialize trajectory log directory (after checkpoint load so global_steps is correct)
        self._init_trajectory_log_dir(resumed_from_step=self.global_steps)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        _traj_buffer: list = []  # accumulates trajectories across sub-batches for one training step
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                _step_start_ts = time.time()
                print(f"[FIT] === step={self.global_steps} epoch={epoch} | loop-top reached ===", flush=True)

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        print(f"[FIT] step={self.global_steps} gen_batch={num_gen_batches} | "
                              f"calling generate_sequences ...", flush=True)
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        print(f"[FIT] step={self.global_steps} gen_batch={num_gen_batches} | "
                              f"generate_sequences done, elapsed={time.time()-_step_start_ts:.1f}s", flush=True)

                    # Decode training inputs/outputs now; scores appended after reward computation
                    try:
                        _train_input_ids = new_batch.batch.get("input_ids")
                        _train_output_ids = gen_batch_output.batch.get("responses")
                        _train_inputs = (
                            [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in _train_input_ids]
                            if _train_input_ids is not None else []
                        )
                        _train_outputs = (
                            [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in _train_output_ids]
                            if _train_output_ids is not None else []
                        )
                    except Exception as _e:
                        print(f"[TrajectoryLog] Warning: failed to decode train trajectories at step "
                              f"{self.global_steps}: {_e}")
                        _train_inputs, _train_outputs = [], []

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        print(f"[FIT] step={self.global_steps} gen_batch={num_gen_batches} | "
                              f"computing reward ...", flush=True)
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # ---- Accumulate training trajectories into buffer ----
                    if _train_inputs and hasattr(self, "_traj_log_base_dir"):
                        try:
                            _n_orig = len(_train_inputs)
                            _n_repeat = self.config.actor_rollout_ref.rollout.n
                            _all_scores = new_batch.batch["token_level_scores"].sum(dim=-1).cpu().tolist()
                            for orig_i in range(_n_orig):
                                for rep_j in range(_n_repeat):
                                    flat_idx = orig_i * _n_repeat + rep_j
                                    inp = _train_inputs[orig_i]
                                    try:
                                        resp_ids = new_batch.batch["responses"][flat_idx]
                                        out = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                                    except Exception:
                                        out = _train_outputs[orig_i] if orig_i < len(_train_outputs) else ""
                                    sc = _all_scores[flat_idx] if flat_idx < len(_all_scores) else 0.0
                                    extra_fields = {}
                                    if reward_extra_infos_dict:
                                        for ek, ev in reward_extra_infos_dict.items():
                                            extra_fields[ek] = ev[flat_idx] if flat_idx < len(ev) else None
                                    _traj_buffer.append({"input": inp, "output": out, "score": sc, **extra_fields})
                        except Exception as _e:
                            print(f"[TrajectoryLog] Warning: failed to accumulate train trajectories at step "
                                  f"{self.global_steps}: {_e}")

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "acc":
                            new_batch.non_tensor_batch["acc"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        # stats: count degenerate 0/1 groups
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)
                        num_all_zero_groups = 0
                        num_all_one_groups = 0
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            vals = np.array(metric_vals)
                            if np.all(vals == 0):
                                num_all_zero_groups += 1
                            if np.all(vals == 1):
                                num_all_one_groups += 1
                        total_groups = len(prompt_uid2metric_vals)
                        print(f"[no_filter] gen_batch {num_gen_batches}: "
                              f"total_groups={total_groups}, "
                              f"all_zero_groups={num_all_zero_groups}, "
                              f"all_one_groups={num_all_one_groups}")
                        metrics["train/all_zero_groups"] = num_all_zero_groups
                        metrics["train/all_one_groups"] = num_all_one_groups

                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "acc":
                            new_batch.non_tensor_batch["acc"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # stats: count degenerate 0/1 groups before filtering
                        num_all_zero_groups = 0
                        num_all_one_groups = 0
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            vals = np.array(metric_vals)
                            if np.all(vals == 0):
                                num_all_zero_groups += 1
                            if np.all(vals == 1):
                                num_all_one_groups += 1
                        total_groups = len(prompt_uid2metric_vals)
                        print(f"[filter_groups] gen_batch {num_gen_batches}: "
                              f"total_groups={total_groups}, "
                              f"all_zero_groups={num_all_zero_groups}, "
                              f"all_one_groups={num_all_one_groups}")
                        metrics["train/all_zero_groups"] = num_all_zero_groups
                        metrics["train/all_one_groups"] = num_all_one_groups

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"[FIT] step={self.global_steps} gen_batch={num_gen_batches} | "
                                  f"{num_prompt_in_batch=} < {prompt_bsz=}, need more generations", flush=True)
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # ---- Flush accumulated training trajectories to disk for this step ----
                    if _traj_buffer and hasattr(self, "_traj_log_base_dir"):
                        try:
                            out_path = os.path.join(
                                self._traj_log_base_dir, "train", f"step_{self.global_steps:06d}.jsonl"
                            )
                            with open(out_path, "w", encoding="utf-8") as _f:
                                for _i, _rec in enumerate(_traj_buffer):
                                    _record = {
                                        "step": self.global_steps,
                                        "idx": _i,
                                        "input": _rec.pop("input"),
                                        "output": _rec.pop("output"),
                                        "score": _rec.pop("score"),
                                        **_rec,
                                    }
                                    _f.write(json.dumps(_record, ensure_ascii=False) + "\n")
                            print(f"[TrajectoryLog] Saved {len(_traj_buffer)} trajectories "
                                  f"({num_gen_batches} sub-batches) → {out_path}")
                        except Exception as _e:
                            print(f"[TrajectoryLog] Warning: failed to flush train trajectories at step "
                                  f"{self.global_steps}: {_e}")

                    print(f"[FIT] step={self.global_steps} | calling sleep_replicas ...", flush=True)
                    self.checkpoint_manager.sleep_replicas()
                    print(f"[FIT] step={self.global_steps} | sleep_replicas done, "
                          f"elapsed={time.time()-_step_start_ts:.1f}s", flush=True)

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        print(f"[FIT] step={self.global_steps} | computing kl_related_metrics ...", flush=True)
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)
                        print(f"[FIT] step={self.global_steps} | kl_related_metrics done, "
                              f"elapsed={time.time()-_step_start_ts:.1f}s", flush=True)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            print(f"[FIT] step={self.global_steps} | calling update_actor ...", flush=True)
                            actor_output = self._update_actor(batch)
                            print(f"[FIT] step={self.global_steps} | update_actor done, "
                                  f"elapsed={time.time()-_step_start_ts:.1f}s", flush=True)

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            print(f"[FIT] step={self.global_steps} | calling update_weights ...", flush=True)
                            self.checkpoint_manager.update_weights()
                            print(f"[FIT] step={self.global_steps} | update_weights done, "
                                  f"elapsed={time.time()-_step_start_ts:.1f}s", flush=True)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        print(f"[FIT] step={self.global_steps} | starting validation ...", flush=True)
                        val_metrics: dict = self._validate()
                        print(f"[FIT] step={self.global_steps} | validation done, "
                              f"elapsed={time.time()-_step_start_ts:.1f}s", flush=True)
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                    # Keep only the best-3 checkpoints by AIME pass@1 plus the current one.
                    self._record_and_prune_checkpoints(val_metrics, keep_best=3)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                _traj_buffer = []  # reset for next step

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
