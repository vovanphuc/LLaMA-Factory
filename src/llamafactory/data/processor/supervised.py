# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            # Mask thinking tokens if requested
            if (self.data_args.enable_thinking and
                self.data_args.mask_thinking_loss and
                hasattr(self.template, 'thought_words')):
                target_label = self._mask_thinking_tokens(target_ids, target_label)

                # Debug: Show word-by-word masking
                print("\n" + "="*80)
                print("DEBUG: Word-by-Word Masking Analysis")
                print("="*80)
                for idx, (token_id, label_id) in enumerate(zip(target_ids, target_label)):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    is_masked = (label_id == IGNORE_INDEX)
                    status = "🚫 MASKED" if is_masked else "✅ LOSS"
                    print(f"[{idx:3d}] {repr(token_text):30s} → {status}")

                masked_count = sum(1 for l in target_label if l == IGNORE_INDEX)
                loss_count = len(target_label) - masked_count
                print(f"\n📊 Summary: {masked_count} masked, {loss_count} with loss (Total: {len(target_label)})")
                print("="*80 + "\n")

                # Debug: Visual side-by-side comparison
                print("="*80)
                print("DEBUG: Visual Comparison (Input IDs vs Labels)")
                print("="*80)

                # Create visual representation
                token_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in target_ids]

                # Find contiguous masked and loss regions
                regions = []
                current_masked = (target_label[0] == IGNORE_INDEX)
                region_start = 0

                for i in range(1, len(target_label)):
                    is_masked = (target_label[i] == IGNORE_INDEX)
                    if is_masked != current_masked:
                        regions.append((region_start, i, current_masked))
                        region_start = i
                        current_masked = is_masked
                # Add final region
                regions.append((region_start, len(target_label), current_masked))

                # Print input_ids line
                print("input_ids:  [", end="")
                for i, token_text in enumerate(token_texts):
                    if i > 0:
                        print(", ", end="")
                    # Truncate long tokens
                    display_text = token_text.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 15:
                        display_text = display_text[:12] + "..."
                    print(f"{display_text}", end="")
                print("]")

                # Print labels line
                print("labels:     [", end="")
                for i, (token_id, label_id) in enumerate(zip(target_ids, target_label)):
                    if i > 0:
                        print(", ", end="")
                    if label_id == IGNORE_INDEX:
                        print("-100", end="")
                    else:
                        token_text = self.tokenizer.decode([label_id], skip_special_tokens=False)
                        display_text = token_text.replace('\n', '\\n').replace('\t', '\\t')
                        if len(display_text) > 15:
                            display_text = display_text[:12] + "..."
                        print(f"{display_text}", end="")
                print("]")

                # Print visual indicators
                print("            ", end="")
                for start, end, is_masked in regions:
                    region_len = sum(len(str(token_texts[i])) for i in range(start, end))
                    region_len += (end - start - 1) * 2  # commas and spaces
                    if start == 0:
                        region_len += 1  # opening bracket
                    else:
                        region_len += 2  # comma and space before region

                    if is_masked:
                        print("^" * min(region_len, 50), end="")
                        print(" NO LOSS ", end="")
                    else:
                        print("^" * min(region_len, 50), end="")
                        print(" LOSS ", end="")
                print()
                print("="*80 + "\n")

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

    def _mask_thinking_tokens(
        self,
        target_ids: list[int],
        target_label: list[int]
    ) -> list[int]:
        """
        Mask thinking tokens in target_label with IGNORE_INDEX.

        This method identifies thinking tokens (enclosed in <think>...</think> tags)
        in the target sequence and replaces their labels with IGNORE_INDEX, effectively
        excluding them from loss computation while keeping them visible in the output.

        Args:
            target_ids: List of token IDs in the target sequence
            target_label: List of label IDs (initially a copy of target_ids)

        Returns:
            Modified target_label with thinking tokens masked (set to IGNORE_INDEX)
        """
        # Get thinking tag strings from template
        # Strip newlines as they may be normalized during processing
        think_start_tag = self.template.thought_words[0].strip()
        think_end_tag = self.template.thought_words[1].strip()

        # Debug: Show what we're looking for
        print("\n" + "="*80)
        print("DEBUG: Thinking Tag Patterns")
        print("="*80)
        print(f"Start tag: {repr(think_start_tag)}")
        print(f"End tag:   {repr(think_end_tag)}")

        # Decode target to find thinking blocks (text-based approach to handle tokenization boundaries)
        target_text = self.tokenizer.decode(target_ids, skip_special_tokens=False)
        print(f"\nFull target sequence: {target_text}")
        print("="*80 + "\n")

        # Find all thinking blocks in the text
        import re
        pattern = re.escape(think_start_tag) + r"(.*?)" + re.escape(think_end_tag)
        matches = list(re.finditer(pattern, target_text, flags=re.DOTALL))

        print(f"DEBUG: Found {len(matches)} thinking block(s) in text")

        # Create mutable copy of labels
        masked_label = list(target_label)

        if len(matches) == 0:
            return masked_label

        # Build character-to-token mapping
        char_to_token = []
        char_pos = 0
        for token_idx, token_id in enumerate(target_ids):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            token_len = len(token_text)
            char_to_token.extend([token_idx] * token_len)
            char_pos += token_len

        print("DEBUG: Scanning for thinking tags...")

        # Mask tokens corresponding to thinking blocks
        for match_idx, match in enumerate(matches):
            start_char = match.start()
            end_char = match.end()

            print(f"  ✓ Match {match_idx + 1}: chars {start_char}-{end_char}")
            print(f"    → Content: {repr(target_text[start_char:end_char][:100])}...")

            # Convert character positions to token positions
            if start_char < len(char_to_token) and end_char <= len(char_to_token):
                start_token_idx = char_to_token[start_char] if start_char < len(char_to_token) else 0
                end_token_idx = char_to_token[end_char - 1] if end_char > 0 and end_char - 1 < len(char_to_token) else len(target_ids) - 1

                print(f"    → Token range: {start_token_idx}-{end_token_idx}")

                # Mask all tokens in this range
                for idx in range(start_token_idx, end_token_idx + 1):
                    if idx < len(masked_label):
                        masked_label[idx] = IGNORE_INDEX

        print()
        return masked_label

    def _match_sequence(
        self,
        ids: list[int],
        start_idx: int,
        pattern: list[int]
    ) -> bool:
        """
        Check if a pattern of token IDs matches the sequence at a given position.

        Args:
            ids: Full sequence of token IDs to search in
            start_idx: Position to start matching from
            pattern: Sequence of token IDs to match

        Returns:
            True if pattern matches at start_idx, False otherwise
        """
        # Check bounds
        if start_idx + len(pattern) > len(ids):
            return False

        # Compare sequences
        return ids[start_idx:start_idx + len(pattern)] == pattern


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
