import re

import numpy as np
from PIL import Image as PILImage
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def _strip_think(text: str) -> str:
    """Remove <think>...</think> block — no-op for non-thinking models."""
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE).strip()


def _extract_yesno(text: str) -> str:
    """Extract the last yes/no from text — for thinking models that reason before answering."""
    text = _strip_think(text)
    matches = list(re.finditer(r'\b(yes|no)\b', text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).lower()
    return text.strip()


class Model:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        max_images: int = 16,
        tensor_parallel_size: int = 1,
        video_mode: bool = False,
        max_model_len: int | None = None,
        max_image_size: int | None = None,
        thinking: bool = False,
    ):
        self._video_mode = video_mode
        self._max_image_size = max_image_size
        # Thinking models must generate freely — constrained sampling blocks <think>.
        self._thinking = thinking
        mm_limits = {"image": max_images, "video": 0}
        if video_mode:
            mm_limits = {"image": 0, "video": 1}
        extra_kwargs = {}
        if video_mode and max_model_len is None:
            # Video mode uses a 262K default context that exceeds available KV cache.
            extra_kwargs["max_model_len"] = 32768
        elif max_model_len is not None:
            extra_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=mm_limits,
            **extra_kwargs,
            # Avoid the outlines backend; its diskcache dep imports sqlite3,
            # which fails on this cluster (undefined symbol: sqlite3_deserialize).
            guided_decoding_backend="lm-format-enforcer",
            # Some models (InternVL) ship custom code in their HF repo
            # and refuse to load without this flag.
            trust_remote_code=True,
        )
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except AttributeError:
            # InternVL3.5 uses Qwen2TokenizerFast which lacks start_image_token
            # expected by InternVLProcessor. The tokenizer alone is sufficient since
            # vLLM handles image processing internally.
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model_id_lower = model_id.lower()

    def _messages(self, images: list[PILImage.Image], system_prompt: str, user_text: str | None = None) -> list[dict]:
        # InternVL expects a plain string with <image>\n per image.
        if "intern" in self._model_id_lower:
            text = "<image>\n" * len(images) + (user_text or "")
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        # Video mode: a single <video> token instead of N <image> tokens.
        if self._video_mode:
            content: list[dict] = [{"type": "video"}]
            if user_text:
                content.append({"type": "text", "text": user_text})
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        content = [{"type": "image", "image": img} for img in images]
        if user_text:
            content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _maybe_resize(self, images: list[PILImage.Image]) -> list[PILImage.Image]:
        if self._max_image_size is None:
            return images
        out = []
        for img in images:
            w, h = img.size
            if max(w, h) > self._max_image_size:
                scale = self._max_image_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), PILImage.BILINEAR)
            out.append(img)
        return out

    def _frames_to_numpy(self, images: list[PILImage.Image]) -> np.ndarray:
        # Stack into (n_frames, H, W, 3) — vllm treats this as one video clip,
        # not as multiple separate videos.
        return np.stack([np.array(img.convert("RGB")) for img in images], axis=0)

    def _prompt(self, messages: list[dict], images: list[PILImage.Image]) -> dict:
        # Molmo uses its own processor.process(text, images) directly — it does
        # not support apply_chat_template. Pass the text as a plain string.
        if "molmo" in self._model_id_lower:
            images = self._maybe_resize(images)
            user_msg = next(m for m in messages if m["role"] == "user")
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            text = ""
            if system_msg:
                text = system_msg["content"] + " "
            content = user_msg["content"]
            text += content if isinstance(content, str) else ""
            inputs = self.processor.process(text=text, images=images)
            return {
                "prompt_token_ids": inputs["input_ids"][0].tolist(),
                "multi_modal_data": {"image": images},
            }
        # Video mode: pass frames as a single video clip to activate 3D RoPE
        # temporal position encodings (instead of treating each frame as an
        # independent image). fps=1 assigns timestamp 0,1,2,... seconds.
        if self._video_mode:
            frames = self._frames_to_numpy(images)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Qwen3-VL's MultiModalDataParser has video_needs_metadata=True,
            # so each video must be a (frames_array, metadata_dict) tuple.
            n = len(images)
            video_metadata = {
                "fps": 1.0,
                "duration": float(n),
                "total_num_frames": n,
                "frames_indices": list(range(n)),
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            return {
                "prompt": prompt,
                "multi_modal_data": {"video": (frames, video_metadata)},
            }
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "multi_modal_data": {"image": images}}

    def ask(self, images: list[PILImage.Image], system_prompt: str, user_text: str | None = None, pattern: str | None = None) -> str:
        return self.ask_batch([(images, system_prompt, user_text)], pattern=pattern)[0]

    def ask_batch(self, requests: list[tuple[list[PILImage.Image], str, str | None]], pattern: str | None = None) -> list[str]:
        # Thinking models generate freely — constrained sampling blocks reasoning.
        # Use a high max_tokens budget so the model can finish its chain-of-thought.
        if self._thinking:
            sampling_params = SamplingParams(max_tokens=4096)
        elif pattern:
            sampling_params = SamplingParams(structured_outputs=StructuredOutputsParams(regex=pattern))
        else:
            sampling_params = None
        inputs = [self._prompt(self._messages(imgs, sp, user_text), imgs) for imgs, sp, user_text in requests]
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        postprocess = _extract_yesno if self._thinking else _strip_think
        return [postprocess(out.outputs[0].text) for out in outputs]
