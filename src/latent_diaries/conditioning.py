"""Continuous conditioning for qwen-tts 0.1.1, non-streaming x-vector path.

No tokenizer or variable text is accepted here. Fixed assistant-role IDs must
come from the pinned tokenizer. ICL reference transcripts are not supported.
"""

from dataclasses import dataclass

import torch


@dataclass
class VisualCondition:
    soft_tokens: torch.Tensor
    attention_mask: torch.Tensor
    speaker_embedding: torch.Tensor


class SoftConditioningBuilder:
    def __init__(self, model, role_ids: tuple[int, int, int]):
        self.model = model
        self.role_ids = role_ids
        if len(role_ids) != 3:
            raise ValueError("The pinned Qwen assistant role contains three fixed IDs")

    def build_prefill(self, condition: VisualCondition) -> dict:
        talker = self.model.talker
        config = self.model.config
        codec = config.talker_config
        soft = condition.soft_tokens
        weights = talker.get_input_embeddings().weight
        if soft.device != weights.device:
            raise ValueError("Condition and Talker must share a device")
        soft = soft.to(dtype=weights.dtype)
        if soft.ndim != 3 or soft.shape[2] != codec.hidden_size:
            raise ValueError("soft_tokens must have shape [batch, K, Talker hidden_size]")
        if soft.shape[1] < 1 or not torch.isfinite(soft).all():
            raise ValueError("soft_tokens must be nonempty and finite")
        mask = condition.attention_mask
        if mask.shape != soft.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean [batch, K]")
        if not mask.any(dim=1).all():
            raise ValueError("Each example needs a valid continuous token")
        speaker = condition.speaker_embedding.to(dtype=weights.dtype)
        if speaker.shape != (soft.shape[0], soft.shape[2]):
            raise ValueError("speaker_embedding must be [batch, hidden_size]")
        if mask.device != soft.device or speaker.device != soft.device:
            raise ValueError("Condition tensors must share a device")

        def text(ids):
            ids = torch.tensor([ids], device=soft.device, dtype=torch.long)
            return talker.text_projection(talker.get_text_embeddings()(ids))

        def audio(ids):
            return talker.get_input_embeddings()(
                torch.tensor([ids], device=soft.device, dtype=torch.long)
            )

        bos, eos, pad = text(
            [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]
        ).chunk(3, dim=1)
        role = text(list(self.role_ids))
        protocol = audio(
            [
                codec.codec_think_id,
                codec.codec_think_bos_id,
                codec.codec_language_id["italian"],
                codec.codec_think_eos_id,
            ]
        )
        batch = soft.shape[0]
        codec_prefix = torch.cat(
            [
                protocol.expand(batch, -1, -1),
                speaker[:, None],
                audio([codec.codec_pad_id]).expand(batch, -1, -1),
            ],
            dim=1,
        )
        text_prefix = torch.cat(
            [pad.expand(batch, codec_prefix.shape[1] - 1, -1), bos.expand(batch, -1, -1)], dim=1
        )
        start = torch.cat([role.expand(batch, -1, -1), text_prefix + codec_prefix], dim=1)
        # Compact each variable prefix, then left-pad exactly as upstream generate.
        sequences = []
        for i in range(batch):
            variable = torch.cat([soft[i : i + 1, mask[i]], eos], dim=1)
            end = pad + audio([codec.codec_bos_id])
            sequences.append(
                torch.cat(
                    [start[i : i + 1], variable + audio([codec.codec_pad_id]), end], dim=1
                ).squeeze(0)
            )
        lengths = torch.tensor([len(s) for s in sequences], device=soft.device)
        reversed_sequences = [s.flip(0) for s in sequences]
        embeds = torch.nn.utils.rnn.pad_sequence(reversed_sequences, batch_first=True).flip(1)
        attention = torch.arange(embeds.shape[1], device=soft.device)[None] >= (
            embeds.shape[1] - lengths[:, None]
        )
        return {
            "inputs_embeds": embeds,
            "attention_mask": attention.long(),
            "trailing_text_hidden": pad.expand(batch, -1, -1),
            "tts_pad_embed": pad.expand(batch, -1, -1),
        }

    def build_teacher_forced(self, condition, gold_codec_codes, frame_mask=None) -> dict:
        prefill = self.build_prefill(condition)
        talker = self.model.talker
        codec = self.model.config.talker_config
        codes = gold_codec_codes
        if (
            codes.ndim != 3
            or codes.shape[0] != condition.soft_tokens.shape[0]
            or (codes.shape[2] != codec.num_code_groups)
            or codes.dtype != torch.long
        ):
            raise ValueError("Gold codes must be int64 [batch, frames, num_code_groups]")
        if codes.device != condition.soft_tokens.device:
            raise ValueError("Gold codes and condition must share a device")
        if frame_mask is None:
            frame_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=codes.device)
        if frame_mask.shape != codes.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be boolean [batch, frames]")
        if frame_mask.device != codes.device:
            raise ValueError("Frame mask and codes must share a device")
        if not frame_mask.any(dim=1).all() or (frame_mask[:, 1:] & ~frame_mask[:, :-1]).any():
            raise ValueError("Gold frame masks must be nonempty and right-padded")
        sequences, labels = [], []
        for i in range(codes.shape[0]):
            gold = codes[i, frame_mask[i]]
            audio = talker.get_input_embeddings()(gold[:, 0])
            for j, embed in enumerate(talker.code_predictor.get_input_embeddings()):
                audio = audio + embed(gold[:, j + 1])
            audio = audio + prefill["tts_pad_embed"][i]
            prefix = prefill["inputs_embeds"][i, prefill["attention_mask"][i].bool()]
            stop = (
                talker.get_input_embeddings()(
                    torch.tensor([codec.codec_eos_token_id], device=codes.device)
                )
                + prefill["tts_pad_embed"][i]
            )
            sequences.append(torch.cat([prefix, audio, stop]))
            # HF causal loss shifts labels by one: the codec BOS predicts frame 0.
            labels.append(
                torch.cat(
                    [
                        torch.full((len(prefix),), -100, device=codes.device),
                        gold[:, 0],
                        torch.tensor([codec.codec_eos_token_id], device=codes.device),
                    ]
                )
            )
        inputs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        lengths = torch.tensor([len(s) for s in sequences], device=codes.device)
        attention = torch.arange(inputs.shape[1], device=codes.device)[None] < lengths[:, None]
        return {
            "inputs_embeds": inputs,
            "attention_mask": attention.long(),
            "labels": targets,
            "use_cache": False,
        }

    @torch.no_grad()
    def generate_codes(
        self, condition, max_frames: int, seed: int, *, codec0_greedy: bool = False
    ) -> list[torch.Tensor]:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        codec = self.model.config.talker_config
        if codec.vocab_size <= 1024:
            raise ValueError("Qwen generation reserves the final 1024 protocol IDs")
        prefill = self.build_prefill(condition)
        device = condition.soft_tokens.device
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            result = self.model.talker.generate(
                **prefill,
                max_new_tokens=max_frames + 1,
                min_new_tokens=2,
                do_sample=not codec0_greedy,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
                repetition_penalty=1.05,
                eos_token_id=codec.codec_eos_token_id,
                suppress_tokens=[
                    i
                    for i in range(codec.vocab_size - 1024, codec.vocab_size)
                    if i != codec.codec_eos_token_id
                ],
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        frames = [h[-1] for h in result.hidden_states if h[-1] is not None]
        if not frames:
            return [
                torch.empty(0, codec.num_code_groups, device=device, dtype=torch.long)
                for _ in range(condition.soft_tokens.shape[0])
            ]
        codes = torch.stack(frames, dim=1)
        outputs = []
        for sample in codes:
            stop = (sample[:, 0] == codec.codec_eos_token_id).nonzero()
            length = int(stop[0]) if len(stop) else len(sample)
            outputs.append(sample[: min(length, max_frames)])
        return outputs
