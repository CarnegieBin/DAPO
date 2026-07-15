"""
ExpThink Custom Reward Manager.

Implements Experience-Guided Reward Shaping with Three-Tier Reward:
  - Experience Buffer: tracks shortest correct response length per prompt
  - Three-Tier Reward:
      1.0   if correct AND length <= l*_q * (1 + alpha)   (short & correct)
      r_pen if correct AND length >  l*_q * (1 + alpha)   (long but correct)
      0.0   if incorrect

Buffer scope is controlled by ``expthink_batch``:
  - False (default): a persistent GLOBAL buffer shared across all training steps
    (l*_q only ever decreases over the whole run).
  - True: a per-batch buffer. l*_q is the shortest correct length observed within
    the CURRENT batch only, with no state carried across steps.

    NOTE: per-batch mode groups a prompt's rollouts by prompt string within the
    chunk handed to each reward worker. To make l*_q exact, ensure all rollouts of
    a prompt land in the same chunk by setting ``reward.num_workers=1`` (otherwise
    ``data.chunk()`` may split a prompt group across workers, yielding a slightly
    larger — i.e. more conservative — l*_q for boundary groups).
"""

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from src.custom_think_rm import verify_think_rm


# ============ Experience Buffer (global, shared across calls) ============
# key: prompt_str -> value: shortest correct response token length seen so far
EXPERIENCE_BUFFER: dict[str, float] = {}


@register("expthink")
class ExpThinkRewardManager(RewardManagerBase):
    """ExpThink Reward Manager with Experience-Guided Three-Tier Reward Shaping."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = verify_think_rm
        self.is_async_reward_score = False

        reward_kwargs = config.reward.get("reward_kwargs", {})
        self.alpha = reward_kwargs.get("expthink_alpha", 0.1)
        self.r_pen = reward_kwargs.get("expthink_r_pen", 0.5)
        self.l_max = reward_kwargs.get("expthink_l_max", 16384)
        # When True, use a per-batch buffer instead of the persistent global one.
        self.batch = reward_kwargs.get("expthink_batch", False)

        # Signals RewardLoopWorker.compute_score_batch to route through run_batch,
        # which performs the two-pass (collect -> reduce -> score) per-batch logic.
        self.batch_level = self.batch

    def _three_tier_reward(self, acc, valid_response_length, l_star) -> float:
        """Map (acc, length, l*) to the three-tier reward."""
        if acc <= 0:
            return 0.0
        threshold = l_star * (1 + self.alpha)
        return 1.0 if valid_response_length <= threshold else self.r_pen

    async def _score_item(self, data_item) -> dict:
        """Decode a single item and compute its accuracy / length.

        Returns a dict with the fields needed by both the global (run_single) and
        the per-batch (run_batch) code paths. Does NOT apply the experience buffer.
        """
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        )
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch["data_source"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            acc = result.get("acc", result["score"])
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            acc = result
            reward_extra_info["acc"] = acc

        reward_extra_info["response_length"] = valid_response_length

        return {
            "prompt_str": prompt_str,
            "acc": acc,
            "valid_response_length": valid_response_length,
            "reward_extra_info": reward_extra_info,
        }

    async def run_single(self, data: DataProto) -> dict:
        """Single-item path using the persistent GLOBAL experience buffer."""
        global EXPERIENCE_BUFFER

        assert len(data) == 1, "Only support single data item"
        scored = await self._score_item(data[0])

        acc = scored["acc"]
        valid_response_length = scored["valid_response_length"]

        # Global Experience Buffer lookup and update
        prompt_key = scored["prompt_str"]
        old_l_star = EXPERIENCE_BUFFER.get(prompt_key, self.l_max)

        if acc > 0:
            new_l_star = min(old_l_star, valid_response_length)
            EXPERIENCE_BUFFER[prompt_key] = new_l_star
        else:
            new_l_star = old_l_star

        reward = self._three_tier_reward(acc, valid_response_length, new_l_star)
        return {"reward_score": reward, "reward_extra_info": scored["reward_extra_info"]}

    async def run_batch(self, data: DataProto) -> list[dict]:
        """Per-batch path: buffer scoped to the current batch only.

        Two passes:
          1) Score every item (decode + acc + length), grouping by prompt string.
          2) l*_q = shortest correct length within the batch per prompt (l_max if
             no correct sample); apply the three-tier reward.
        No state is carried across batches.
        """
        import asyncio

        scored = await asyncio.gather(*[self._score_item(data[i]) for i in range(len(data))])

        # Pass 2 reduction: shortest correct length per prompt within this batch
        l_star_by_prompt: dict[str, float] = {}
        for item in scored:
            if item["acc"] > 0:
                key = item["prompt_str"]
                length = item["valid_response_length"]
                if length < l_star_by_prompt.get(key, self.l_max):
                    l_star_by_prompt[key] = length

        outputs = []
        for item in scored:
            l_star = l_star_by_prompt.get(item["prompt_str"], self.l_max)
            reward = self._three_tier_reward(item["acc"], item["valid_response_length"], l_star)
            outputs.append({"reward_score": reward, "reward_extra_info": item["reward_extra_info"]})
        return outputs
