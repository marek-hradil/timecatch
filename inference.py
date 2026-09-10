import re
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams


@dataclass
class Prompt:
    """One chat request: an image sequence, a system prompt, and optional
    trailing user text (e.g. a scene description appended after the images).
    """

    images: list[PILImage.Image]
    system_prompt: str
    user_text: str | None = None

    def intern_messages(self) -> list[dict]:
        # InternVL expects a plain string with <image>\n per image.
        text = "<image>\n" * len(self.images) + (self.user_text or "")
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

    def video_messages(self) -> list[dict]:
        # Video mode: a single <video> token instead of N <image> tokens.
        content: list[dict] = [{"type": "video"}]
        if self.user_text:
            content.append({"type": "text", "text": self.user_text})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    def image_messages(self) -> list[dict]:
        content = [{"type": "image", "image": img} for img in self.images]
        if self.user_text:
            content.append({"type": "text", "text": self.user_text})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]


def build_messages(
    prompt: Prompt, model_id: str, video_mode: bool = False
) -> list[dict]:
    """Pick the message format for `model_id`/`video_mode` and build it.

    Called once per Prompt, by the caller, before invoking Model.ask/ask_batch
    — so the family check lives in exactly one place instead of being copied
    into every calling script, without Model needing to own it either.
    """
    if "intern" in model_id.lower():
        return prompt.intern_messages()
    if video_mode:
        return prompt.video_messages()
    return prompt.image_messages()


def _strip_think(text: str) -> str:
    """Remove <think>...</think> block — no-op for non-thinking models."""
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE).strip()


def _extract_yesno(text: str) -> str:
    """Extract the last yes/no from text — for thinking models that reason before answering."""
    text = _strip_think(text)
    matches = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
    return matches[-1].group(1).lower() if matches else text.strip()


def _frames_to_numpy(images: list[PILImage.Image]) -> np.ndarray:
    # Stack into (n_frames, H, W, 3) — vllm treats this as one video clip,
    # not as multiple separate videos.
    return np.stack([np.array(img.convert("RGB")) for img in images], axis=0)


def _encode_video(
    processor, messages: list[dict], images: list[PILImage.Image]
) -> dict:
    # Pass frames as a single video clip to activate 3D RoPE temporal position
    # encodings (instead of treating each frame as an independent image).
    # fps=1 assigns timestamp 0,1,2,... seconds.
    frames = _frames_to_numpy(images)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    n = len(images)
    # Qwen3-VL's MultiModalDataParser has video_needs_metadata=True,
    # so each video must be a (frames_array, metadata_dict) tuple.
    video_metadata = {
        "fps": 1.0,
        "duration": float(n),
        "total_num_frames": n,
        "frames_indices": list(range(n)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return {"prompt": text, "multi_modal_data": {"video": (frames, video_metadata)}}


def _encode_default(
    processor, messages: list[dict], images: list[PILImage.Image]
) -> dict:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text, "multi_modal_data": {"image": images}}


class Model:
    """Thin vLLM wrapper. Takes already-built chat messages (see
    build_messages()) plus the raw images, and runs constrained — or, for
    thinking models, free-form — generation.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        max_images: int = 16,
        tensor_parallel_size: int = 1,
        video_mode: bool = False,
        max_model_len: int | None = None,
        thinking: bool = False,
        enforce_eager: bool = False,
        lora_path: str | None = None,
        lora_rank: int = 16,
        multi_lora: bool = False,
    ):
        # Thinking models must generate freely — constrained sampling blocks <think>.
        self._thinking = thinking
        self._video_mode = video_mode

        self._next_lora_id = 1
        self._lora_request = (
            self._make_lora_request(lora_path) if lora_path is not None else None
        )

        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=self._mm_limits(max_images, video_mode),
            # Some models (InternVL) ship custom code in their HF repo
            # and refuse to load without this flag.
            trust_remote_code=True,
            **self._llm_kwargs(
                video_mode,
                max_model_len,
                enforce_eager,
                lora_path,
                lora_rank,
                multi_lora,
            ),
        )
        self.processor = self._load_processor(model_id)

    # -- construction helpers --------------------------------------------------

    @staticmethod
    def _mm_limits(max_images: int, video_mode: bool) -> dict:
        if video_mode:
            return {"image": 0, "video": 1, "audio": 0}
        return {"image": max_images, "video": 0, "audio": 0}

    @staticmethod
    def _llm_kwargs(
        video_mode: bool,
        max_model_len: int | None,
        enforce_eager: bool,
        lora_path: str | None,
        lora_rank: int,
        multi_lora: bool,
    ) -> dict:
        kwargs = {}
        if video_mode and max_model_len is None:
            # Video mode uses a 262K default context that exceeds available KV cache.
            kwargs["max_model_len"] = 32768
        elif max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if enforce_eager:
            kwargs["enforce_eager"] = True
        # multi_lora=True enables the vLLM LoRA runtime without pinning a single
        # adapter, so ask_batch(..., lora_path=...) can swap adapters per call
        # against one running model instead of reloading the base model per checkpoint.
        if lora_path is not None or multi_lora:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = lora_rank
        return kwargs

    def _make_lora_request(self, lora_path: str) -> LoRARequest:
        request = LoRARequest("adapter", self._next_lora_id, lora_path)
        self._next_lora_id += 1
        return request

    @staticmethod
    def _load_processor(model_id: str):
        try:
            return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except AttributeError:
            # InternVL3.5 uses Qwen2TokenizerFast which lacks start_image_token
            # expected by InternVLProcessor. The tokenizer alone is sufficient since
            # vLLM handles image processing internally.
            return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # -- public API ---------------------------------------------------------------

    def ask(
        self,
        messages: list[dict],
        images: list[PILImage.Image],
        pattern: str | None = None,
        lora_path: str | None = None,
    ) -> str:
        return self.ask_batch(
            [(messages, images)], pattern=pattern, lora_path=lora_path
        )[0]

    def ask_batch(
        self,
        requests: list[tuple[list[dict], list[PILImage.Image]]],
        pattern: str | None = None,
        lora_path: str | None = None,
    ) -> list[str]:
        lora_request = self._resolve_lora(lora_path)
        sampling_params = self._sampling_params(pattern)
        inputs = []
        for messages, images in requests:
            if self._video_mode:
                inputs.append(_encode_video(self.processor, messages, images))
            else:
                inputs.append(_encode_default(self.processor, messages, images))
        outputs = self.llm.generate(
            inputs, sampling_params=sampling_params, lora_request=lora_request
        )
        postprocess = _extract_yesno if self._thinking else _strip_think
        return [postprocess(out.outputs[0].text) for out in outputs]

    def _resolve_lora(self, lora_path: str | None) -> LoRARequest | None:
        # Per-call override — lets one running model (built with multi_lora=True)
        # swap adapters between calls instead of reloading per checkpoint.
        if lora_path is None:
            return self._lora_request
        request = LoRARequest(
            f"adapter-{self._next_lora_id}", self._next_lora_id, lora_path
        )
        self._next_lora_id += 1
        return request

    def _sampling_params(self, pattern: str | None) -> SamplingParams | None:
        # Thinking models generate freely — constrained sampling blocks reasoning.
        # Use a high max_tokens budget so the model can finish its chain-of-thought.
        if self._thinking:
            return SamplingParams(max_tokens=4096)
        if pattern:
            return SamplingParams(
                structured_outputs=StructuredOutputsParams(regex=pattern)
            )
        return None
