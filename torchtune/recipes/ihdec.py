#!/usr/bin/env python3
"""
IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies.

Per-step delta computation:
1. seq_full = prompt_full + generated_so_far
2. dist_full, dist_no_sys, dist_no_usr, dist_no_asst, dist_no_usr_asst via attention
   column-blocking (batched, no padding needed — all variants share the same token sequence).
3. influence(role) = JSD(dist_full || dist_no_role)
4. Detect hierarchy violations: cs_sys, cs_usr
5. Δ_H = (dist_no_CS(H) - dist_no_H); Δ_final = Σ Δ_H (normalized)

Generation (1 batched forward per step):
  logit_final = dist_full + β × decay^t × Δ_final

Usage:
    CUDA_VISIBLE_DEVICES=0 python ihdec.py --beta 1300 --beta-decay 0.97
    # For Qwen models, --device auto is required:
    CUDA_VISIBLE_DEVICES=0 python ihdec.py --model-type qwen --device auto --beta 1300 --beta-decay 0.97
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchtune.data import Message


# ---------------------------------------------------------------------------
# Role position scanning (one-shot, model-agnostic)
# ---------------------------------------------------------------------------

def _find_role_positions_llama(
    token_ids: List[int],
    start_hdr_id: int,
    end_hdr_id: int,
    eot_id: int,
    role_toks_map: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    """Llama 3 chat format: <|start_header_id|>ROLE<|end_header_id|>\\n\\nCONTENT<|eot_id|>"""
    result: Dict[str, List[int]] = {"system": [], "user": [], "assistant": []}
    ids = token_ids
    n = len(ids)
    i = 0
    while i < n:
        if ids[i] != start_hdr_id:
            i += 1
            continue
        j = i + 1
        while j < n and ids[j] != end_hdr_id:
            j += 1
        if j >= n:
            break
        role_chunk = ids[i + 1 : j]
        matched_role = next(
            (name for name, rtoks in role_toks_map.items() if role_chunk == rtoks), None
        )
        k = j + 1
        while k < n and ids[k] != eot_id:
            k += 1
        if k >= n:
            break  # generation header — no EOT yet
        if matched_role is not None:
            result[matched_role].extend(range(i, k + 1))
        i = k + 1
    return result


def _find_role_positions_chatml(
    token_ids: List[int],
    im_start_id: int,
    im_end_id: int,
    role_toks_map: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    """ChatML format: <|im_start|>ROLE\\nCONTENT<|im_end|>  (used by Qwen etc.)"""
    result: Dict[str, List[int]] = {"system": [], "user": [], "assistant": []}
    ids = token_ids
    n = len(ids)
    i = 0
    while i < n:
        if ids[i] != im_start_id:
            i += 1
            continue
        matched_role = next(
            (
                name
                for name, rtoks in role_toks_map.items()
                if ids[i + 1 : i + 1 + len(rtoks)] == rtoks
            ),
            None,
        )
        k = i + 1
        while k < n and ids[k] != im_end_id:
            k += 1
        if k >= n:
            break  # generation header — no im_end yet
        if matched_role is not None:
            result[matched_role].extend(range(i, k + 1))
        i = k + 1
    return result


# ---------------------------------------------------------------------------
# HuggingFace tokenizer adapter
# ---------------------------------------------------------------------------

class _HFTokenizerAdapter:
    def __init__(self, hf_tokenizer, stop_tokens: Set[int]):
        self._tok = hf_tokenizer
        self._stop_ids = stop_tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        if truncate_at_eos:
            ids: List[int] = []
            for t in token_ids:
                if t in self._stop_ids:
                    break
                ids.append(t)
            token_ids = ids
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)


# ---------------------------------------------------------------------------
# HuggingFace-based model (supports Qwen3, Llama via HF, etc.)
# ---------------------------------------------------------------------------

class HFModel:
    """Drop-in replacement for LocalModel using HuggingFace AutoModel."""

    DEFAULT_MODEL_PATH = "../pretrained_models/qwen3_8b"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        dtype: str = "bf16",
        max_seq_len: int = 4096,
        enable_thinking: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = torch.device("cuda") if device == "auto" else torch.device(device)
        self._dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self._max_seq_len = max_seq_len

        print(f"Loading model from {model_path} ...")
        hf_tok = AutoTokenizer.from_pretrained(model_path)
        device_map = "auto" if device == "auto" else str(self._device)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            device_map=device_map,
        )
        self._model.eval()
        if device == "auto":
            self._device = next(self._model.parameters()).device

        stop_ids: Set[int] = set()
        if hf_tok.eos_token_id is not None:
            stop_ids.add(hf_tok.eos_token_id)
        for tok_name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            tid = hf_tok.convert_tokens_to_ids(tok_name)
            if tid is not None and tid != hf_tok.unk_token_id:
                stop_ids.add(tid)

        self._stop_tokens: Set[int] = stop_ids
        self._hf_tokenizer = hf_tok
        self._tokenizer = _HFTokenizerAdapter(hf_tok, stop_ids)

        path_lower = model_path.lower()
        if "qwen" in path_lower:
            self._chat_format = "chatml"
        elif "llama" in path_lower:
            self._chat_format = "llama"
        else:
            im_start = hf_tok.convert_tokens_to_ids("<|im_start|>")
            self._chat_format = "chatml" if im_start != hf_tok.unk_token_id else "llama"

        self._enable_thinking = enable_thinking
        print(f"Model ready. (chat_format={self._chat_format}, enable_thinking={enable_thinking})")

    def _prompt_tokens(self, messages: List[Message]) -> List[int]:
        _content = (lambda m: m.text_content) if self._chat_format == "chatml" else (lambda m: m.content)
        hf_msgs = [{"role": m.role, "content": _content(m)} for m in messages]
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if self._chat_format == "chatml":
            kwargs["enable_thinking"] = self._enable_thinking
        result = self._hf_tokenizer.apply_chat_template(hf_msgs, **kwargs)
        return result if isinstance(result, list) else result["input_ids"]

    def prompt_tokens_with_roles(
        self, messages: List[Message]
    ) -> Tuple[List[int], Dict[str, List[int]]]:
        _content = (lambda m: m.text_content) if self._chat_format == "chatml" else (lambda m: m.content)
        hf_msgs = [{"role": m.role, "content": _content(m)} for m in messages]
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if self._chat_format == "chatml":
            kwargs["enable_thinking"] = self._enable_thinking
        result = self._hf_tokenizer.apply_chat_template(hf_msgs, **kwargs)
        tokens: List[int] = result if isinstance(result, list) else result["input_ids"]
        role_toks_map = {
            name: self._hf_tokenizer.encode(name, add_special_tokens=False)
            for name in ("system", "user", "assistant")
        }
        if self._chat_format == "chatml":
            im_start = self._hf_tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end = self._hf_tokenizer.convert_tokens_to_ids("<|im_end|>")
            role_pos = _find_role_positions_chatml(tokens, im_start, im_end, role_toks_map)
        else:
            start_hdr = self._hf_tokenizer.convert_tokens_to_ids("<|start_header_id|>")
            end_hdr = self._hf_tokenizer.convert_tokens_to_ids("<|end_header_id|>")
            eot = self._hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            role_pos = _find_role_positions_llama(tokens, start_hdr, end_hdr, eot, role_toks_map)
        return tokens, role_pos

    @torch.inference_mode()
    def _log_probs(self, token_ids: List[int]) -> torch.Tensor:
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        out = self._model(input_ids=tokens, use_cache=False)
        return F.log_softmax(out.logits[0, -1, :].float(), dim=-1)

    @property
    def _is_multi_device(self) -> bool:
        return len({p.device for p in self._model.parameters()}) > 1

    @torch.inference_mode()
    def masked_batch_log_probs(
        self, token_ids: List[int], masks: torch.Tensor
    ) -> List[torch.Tensor]:
        """Batched forward with per-item boolean causal masks [B, L, L]."""
        B, L, _ = masks.shape

        if self._is_multi_device:
            pad_id = (self._hf_tokenizer.pad_token_id
                      or self._hf_tokenizer.eos_token_id or 0)
            results = []
            for b in range(B):
                blocked = (~masks[b, -1, :]).nonzero(as_tuple=True)[0].tolist()
                ablated = list(token_ids)
                for pos in blocked:
                    ablated[pos] = pad_id
                results.append(self._log_probs(ablated))
            return results

        tokens = (
            torch.tensor(token_ids, dtype=torch.long, device=self._device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        attn_mask = torch.zeros(B, 1, L, L, device=self._device, dtype=self._dtype)
        fill_val = torch.finfo(self._dtype).min if self._chat_format == "chatml" else torch.finfo(torch.float32).min
        attn_mask.masked_fill_(~masks.unsqueeze(1), fill_val)

        out = self._model(input_ids=tokens, attention_mask=attn_mask, use_cache=False)
        lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        return [lp[i] for i in range(B)]


# ---------------------------------------------------------------------------
# LocalModel (torchtune / Llama)
# ---------------------------------------------------------------------------

class LocalModel:
    DEFAULT_CHECKPOINT_DIR = "../pretrained_models/llama3_1_8B_Instruct/original"
    DEFAULT_CHECKPOINT_FILE = "consolidated.00.pth"
    DEFAULT_TOKENIZER_PATH = (
        "../pretrained_models/llama3_1_8B_Instruct/original/tokenizer.model"
    )

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        device: str = "cuda",
        dtype: str = "bf16",
        max_seq_len: int = 4096,
    ):
        from torchtune import utils
        from torchtune.models.llama3 import llama3_tokenizer
        from torchtune.models.llama3_1 import llama3_1_8b
        from torchtune.utils import FullModelMetaCheckpointer

        self._device = utils.get_device(device)
        self._dtype = utils.get_dtype(dtype, device=self._device)

        print(f"Loading model from {checkpoint_dir}/{checkpoint_file} ...")
        checkpointer = FullModelMetaCheckpointer(
            checkpoint_dir=checkpoint_dir,
            checkpoint_files=[checkpoint_file],
            recipe_checkpoint=None,
            output_dir=checkpoint_dir,
            model_type="LLAMA3",
        )
        ckpt = checkpointer.load_checkpoint()

        with utils.set_default_dtype(self._dtype), self._device:
            self._model = llama3_1_8b()
        self._model.load_state_dict(ckpt[utils.MODEL_KEY])
        self._model.eval()

        self._model.max_seq_len = max_seq_len
        with self._device:
            self._model.setup_caches(batch_size=1, dtype=self._dtype)

        self._tokenizer = llama3_tokenizer(path=tokenizer_path)
        self._stop_tokens: Set[int] = set(self._tokenizer.stop_tokens)
        self._max_seq_len = max_seq_len
        print("Model ready.")

    def _prompt_tokens(self, messages: List[Message]) -> List[int]:
        tokens, _ = self._tokenizer.tokenize_messages(messages, add_eos=False)
        header = (
            [self._tokenizer.start_header_id]
            + self._tokenizer.encode("assistant", add_bos=False, add_eos=False)
            + [self._tokenizer.end_header_id]
            + self._tokenizer.encode("\n\n", add_bos=False, add_eos=False)
        )
        return tokens + header

    def prompt_tokens_with_roles(
        self, messages: List[Message]
    ) -> Tuple[List[int], Dict[str, List[int]]]:
        tokens = self._prompt_tokens(messages)
        role_toks_map = {
            name: self._tokenizer.encode(name, add_bos=False, add_eos=False)
            for name in ("system", "user", "assistant")
        }
        role_pos = _find_role_positions_llama(
            tokens,
            self._tokenizer.start_header_id,
            self._tokenizer.end_header_id,
            self._tokenizer.eot_id,
            role_toks_map,
        )
        return tokens, role_pos

    @torch.inference_mode()
    def masked_batch_log_probs(
        self, token_ids: List[int], masks: torch.Tensor
    ) -> List[torch.Tensor]:
        """Batched forward with per-item boolean causal masks [B, L, L]."""
        inner = self._model
        B = masks.shape[0]
        L = len(token_ids)
        dev = self._device

        tokens = (
            torch.tensor(token_ids, dtype=torch.int, device=dev)
            .unsqueeze(0)
            .expand(B, -1)
        )
        input_pos = torch.arange(L, device=dev)

        saved_causal_mask = inner.causal_mask
        inner.causal_mask = None
        saved_kv_caches = []
        for layer in inner.layers:
            saved_kv_caches.append(layer.attn.kv_cache)
            layer.attn.kv_cache = None

        try:
            logits = inner(tokens, mask=masks, input_pos=input_pos)
        finally:
            inner.causal_mask = saved_causal_mask
            for layer, kvc in zip(inner.layers, saved_kv_caches):
                layer.attn.kv_cache = kvc

        lp = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        return [lp[i] for i in range(B)]

    @torch.inference_mode()
    def _log_probs(self, token_ids: List[int]) -> torch.Tensor:
        self._model.reset_caches()
        tokens = torch.tensor(token_ids, dtype=torch.int, device=self._device).unsqueeze(0)
        input_pos = torch.arange(len(token_ids), device=self._device)
        logits = self._model(tokens, input_pos=input_pos)
        return F.log_softmax(logits[0, -1, :].float(), dim=-1)


# ---------------------------------------------------------------------------
# JSD
# ---------------------------------------------------------------------------

def _sample(log_probs: torch.Tensor, temperature: float = 0.6, top_k: int = 300) -> int:
    scaled = log_probs / temperature
    top_vals, top_ids = torch.topk(scaled, min(top_k, scaled.size(-1)))
    filtered = torch.full_like(scaled, float("-inf"))
    filtered.scatter_(0, top_ids, top_vals)
    probs = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


def _jsd(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-40).log()
    kl_pm = (p * (log_p - log_m)).sum()
    kl_qm = (q * (log_q - log_m)).sum()
    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# Attention mask building
# ---------------------------------------------------------------------------

def _ablation_mask(seq_len: int, block_positions: List[int], device) -> torch.Tensor:
    """Boolean causal mask [L, L] with specified key columns blocked.

    mask[i, j] = True  → query position i can attend to key position j.
    Causal constraint: j <= i. Blocked columns are set False regardless.
    """
    pos = torch.arange(seq_len, device=device)
    mask = pos.unsqueeze(0) <= pos.unsqueeze(1)  # mask[i, j] = (j <= i)
    if block_positions:
        blocked = torch.tensor(block_positions, dtype=torch.long, device=device)
        mask[:, blocked] = False
    return mask


# ---------------------------------------------------------------------------
# Conversation parsing & message building
# ---------------------------------------------------------------------------

def _parse_conv(conv: List[Dict]) -> Tuple[str, List[Message]]:
    system = next((m["content"] for m in conv if m["role"] == "system"), "")
    turns = [
        Message(role=m["role"], content=m["content"])
        for m in conv if m["role"] in ("user", "assistant")
    ]
    return system, turns


def _build_msgs(
    system: str,
    turns: List[Message],
    remove_roles: Set[str],
) -> List[Message]:
    msgs: List[Message] = []
    if "sys" not in remove_roles and system:
        msgs.append(Message(role="system", content=system))
    for turn in turns:
        if turn.role == "user"      and "usr"  in remove_roles:
            continue
        if turn.role == "assistant" and "asst" in remove_roles:
            continue
        msgs.append(turn)
    return msgs


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class JSDResult:
    response: str
    tokens_generated: int
    influences: Dict[str, float]
    violations: Dict[str, Any]
    delta_norm: float
    prompt_full_len: int
    beta_decay: float


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class IHJSDBiasDecoderV5:
    def __init__(self, model: LocalModel):
        self._model = model

    def generate(
        self,
        conv: List[Dict],
        max_new_tokens: int = 512,
        beta: float = 1.0,
        beta_decay: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[JSDResult, str, List[Message]]:
        result = self._generate(conv, max_new_tokens, beta, beta_decay, verbose)
        system, turns = _parse_conv(conv)
        return result, system, turns

    @torch.inference_mode()
    def _generate(
        self,
        conv: List[Dict],
        max_new_tokens: int = 512,
        beta: float = 1.0,
        beta_decay: float = 1.0,
        verbose: bool = True,
    ) -> JSDResult:
        m = self._model
        log = print if verbose else (lambda *a, **k: None)

        system, turns = _parse_conv(conv)
        has_sys  = bool(system)
        has_asst = any(t.role == "assistant" for t in turns)

        prompt_full, role_pos = m.prompt_tokens_with_roles(_build_msgs(system, turns, remove_roles=set()))
        L_prompt = len(prompt_full)
        dev = m._device

        log(f"  beta={beta}  beta_decay={beta_decay}")
        log(f"  Prompt tokens — full: {L_prompt}")
        log(
            f"  Role positions — sys: {len(role_pos.get('system', []))}, "
            f"usr: {len(role_pos.get('user', []))}, asst: {len(role_pos.get('assistant', []))}"
        )

        # ── Per-step delta computation ─────────────────────────────────────

        def _compute_delta_and_logit(
            generated: List[int],
        ) -> Tuple[Optional[torch.Tensor], Dict, Dict, float, torch.Tensor]:
            """Compute delta and dist_full for seq_full = prompt_full + generated.

            Only computes ablations for roles that are actually present.
            Returns: (delta_final, violations, influences, delta_norm, dist_full)
            """
            seq = prompt_full + generated
            L   = len(seq)

            # Build mask list dynamically based on present roles
            masks_list = [_ablation_mask(L, [], dev)]  # mask_full at index 0
            idx: Dict[str, int] = {}

            if has_sys:
                idx["sys"] = len(masks_list)
                masks_list.append(_ablation_mask(L, role_pos.get("system", []), dev))

            idx["usr"] = len(masks_list)
            masks_list.append(_ablation_mask(L, role_pos.get("user", []), dev))

            if has_asst and role_pos.get("assistant"):
                idx["asst"] = len(masks_list)
                masks_list.append(_ablation_mask(L, role_pos["assistant"], dev))
                if has_sys:
                    idx["usr_asst"] = len(masks_list)
                    masks_list.append(_ablation_mask(L, role_pos.get("user", []) + role_pos["assistant"], dev))

            dists = m.masked_batch_log_probs(seq, torch.stack(masks_list))
            dist_full        = dists[0]
            dist_no_sys      = dists[idx["sys"]]      if "sys"      in idx else None
            dist_no_usr      = dists[idx["usr"]]
            dist_no_asst     = dists[idx["asst"]]     if "asst"     in idx else None
            dist_no_usr_asst = dists[idx["usr_asst"]] if "usr_asst" in idx else None

            # ── Influence ─────────────────────────────────────────────────
            inf_sys  = _jsd(dist_full, dist_no_sys)  if dist_no_sys  is not None else None
            inf_usr  = _jsd(dist_full, dist_no_usr)
            inf_asst = _jsd(dist_full, dist_no_asst) if dist_no_asst is not None else None

            influences: Dict[str, float] = {}
            if inf_sys  is not None: influences["sys"]  = round(inf_sys,  5)
            influences["usr"] = round(inf_usr, 5)
            if inf_asst is not None: influences["asst"] = round(inf_asst, 5)

            # ── Hierarchy violation detection ──────────────────────────────
            cs_sys: Set[str] = set()
            if inf_sys is not None:
                if inf_usr > inf_sys: cs_sys.add("usr")
                if inf_asst is not None and inf_asst > inf_sys: cs_sys.add("asst")

            cs_usr: Set[str] = set()
            if inf_asst is not None and inf_asst > inf_usr:
                cs_usr.add("asst")

            violations = {"cs_sys": sorted(cs_sys), "cs_usr": sorted(cs_usr)}

            # ── Delta computation ──────────────────────────────────────────
            inf_map: Dict[str, float] = {"usr": inf_usr}
            if inf_asst is not None: inf_map["asst"] = inf_asst

            def _get_dist_no_cs(cs: Set[str]) -> Optional[torch.Tensor]:
                if cs == {"usr"}:         return dist_no_usr
                if cs == {"asst"}:        return dist_no_asst
                if cs == {"usr", "asst"}: return dist_no_usr_asst
                return None

            delta_final: Optional[torch.Tensor] = None

            def _accumulate(dist_no_H: Optional[torch.Tensor], cs: Set[str], H_inf: float):
                nonlocal delta_final
                if dist_no_H is None or not cs:
                    return
                dist_no_cs = _get_dist_no_cs(cs)
                if dist_no_cs is None:
                    return
                delta_H = dist_no_cs - dist_no_H
                delta_final = delta_H if delta_final is None else delta_final + delta_H

            if has_sys and inf_sys is not None:
                if cs_sys == {"usr", "asst"}:
                    _accumulate(dist_no_sys, {"usr", "asst"}, inf_sys)
                else:
                    if cs_sys: _accumulate(dist_no_sys, cs_sys, inf_sys)
                    if cs_usr: _accumulate(dist_no_usr, cs_usr, inf_usr)
            else:
                # sys 없음: usr vs asst 계층만 체크
                if cs_usr: _accumulate(dist_no_usr, cs_usr, inf_usr)

            delta_norm: float = 0.0
            if delta_final is not None:
                delta_norm  = delta_final.norm().item()
                delta_final = delta_final / (delta_final.norm() + 1e-8)

            return delta_final, violations, influences, delta_norm, dist_full

        # ── Qwen thinking mode setup ───────────────────────────────────────
        is_qwen = getattr(m, "_chat_format", None) == "chatml"
        enable_thinking = getattr(m, "_enable_thinking", False)
        in_thinking = is_qwen and enable_thinking
        think_end_ids: List[int] = []
        think_end_pos: int = 0  # generated 내 </think> 직후 위치 (Llama는 0 유지)
        if is_qwen and enable_thinking:
            think_end_ids = m._hf_tokenizer.encode("</think>", add_special_tokens=False)
            log(f"  [Qwen] thinking mode active, bias deferred until </think> (ids={think_end_ids})")

        # ── Generation loop ────────────────────────────────────────────────
        generated: List[int] = []
        t = 0

        delta_t:     Optional[torch.Tensor] = None
        violations_t: Dict[str, Any]        = {}
        influences_t: Dict[str, float]      = {}
        norm_t:       float                 = 0.0
        logit:        Optional[torch.Tensor] = None

        first_violations: Dict[str, Any]   = {}
        first_influences: Dict[str, float] = {}
        first_delta_norm: float            = 0.0

        # user만 있는 경우: JSD 계산 없이 순수 greedy
        greedy_only = not has_sys and not has_asst

        log(f"  --- Generation{', greedy-only' if greedy_only else ''} ---")

        while t < max_new_tokens:
            if L_prompt + len(generated) >= m._max_seq_len:
                break

            if in_thinking or greedy_only:
                # thinking 구간 또는 role이 1개: bias 없이 단일 pass만
                logit = m._log_probs(prompt_full + generated)
            else:
                # 배치 pass: 모든 variant + main logit 동시 계산
                delta_t, violations_t, influences_t, norm_t, logit = \
                    _compute_delta_and_logit(generated)
                if t == 0:
                    log(f"  step0 influences: {influences_t}  violations: {violations_t}  delta_norm={norm_t:.4f}")
                    if delta_t is None:
                        log(f"  No violations — standard greedy decoding.")

            if t == 0 and not in_thinking:
                first_violations = violations_t
                first_influences = influences_t
                first_delta_norm = norm_t

            if delta_t is not None and not in_thinking:
                logit = logit + (beta * beta_decay ** t) * delta_t

            if greedy_only:
                next_tok = _sample(logit, temperature=0.6, top_k=300)
            else:
                next_tok = int(logit.argmax())
            generated.append(next_tok)

            if in_thinking:
                n = len(think_end_ids)
                if len(generated) >= n and generated[-n:] == think_end_ids:
                    in_thinking = False
                    t = 0
                    delta_t = None
                    think_end_pos = len(generated)
                    log(f"  [Qwen] </think> detected (pos={think_end_pos}), bias now active.")
            else:
                t += 1

            if next_tok in m._stop_tokens:
                break

        if is_qwen:
            text = m._tokenizer.decode(generated[think_end_pos:], truncate_at_eos=True, skip_special_tokens=True)
        else:
            text = m._tokenizer.decode(generated, truncate_at_eos=True, skip_special_tokens=True)
        log(f"  → {text}")

        return JSDResult(
            response=text,
            tokens_generated=t,
            influences=first_influences,
            violations=first_violations,
            delta_norm=first_delta_norm,
            prompt_full_len=L_prompt,
            beta_decay=beta_decay,
        )


# ---------------------------------------------------------------------------
# Batch file processing
# ---------------------------------------------------------------------------

def process_file(
    decoder: IHJSDBiasDecoderV5,
    input_path: Path,
    output_path: Path,
    max_new_tokens: int = 512,
    beta: float = 1.0,
    beta_decay: float = 1.0,
    n: Optional[int] = None,
    verbose: bool = True,
) -> None:
    with open(input_path, encoding="utf-8") as f:
        conversations: List[Any] = json.load(f)
    if n is not None:
        conversations = conversations[:n]

    print(f"\n{'=' * 64}")
    print(f"Processing : {input_path.name}  ({len(conversations)} conversations)")
    print(f"Output     : {output_path}")
    print(f"beta={beta}  beta_decay={beta_decay}")
    print(f"{'=' * 64}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Any] = []
    done_ids: set = set()
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text(encoding="utf-8"))
            done_ids = {r["id"] for r in results}
            print(f"  [resume] {len(done_ids)} already done, skipping.")
        except Exception:
            results = []

    n_params = sum(p.numel() for p in decoder._model._model.parameters())

    for idx, conv in enumerate(conversations):
        if idx in done_ids:
            continue
        print(f"\n--- Conversation {idx + 1}/{len(conversations)} ---", flush=True)
        try:
            _t0 = time.time()
            result, system, turns = decoder.generate(
                conv, max_new_tokens, beta, beta_decay, verbose=False,
            )
            _elapsed = time.time() - _t0
            _avg_seq  = result.prompt_full_len + result.tokens_generated / 2
            _b = 5 if any(t.role == "assistant" for t in turns) else 3
            _gflops = 2 * n_params * _b * result.tokens_generated * _avg_seq / 1e9
            print(f"  influences      : {result.influences}")
            print(f"  violations      : {result.violations}")
            print(f"  delta_norm      : {result.delta_norm:.4f}")
            print(f"  tokens_generated: {result.tokens_generated}")
            print(f"  inference_time  : {_elapsed:.2f}s")
            print(f"  latency         : {_elapsed / max(result.tokens_generated, 1) * 1000:.1f} ms/tok  |  {result.tokens_generated / max(_elapsed, 1e-9):.1f} tok/s")
            print(f"  est. GFLOPs     : {_gflops:.1f}  |  {_gflops / max(_elapsed, 1e-9):.1f} GFLOP/s")
            print(f"  → {result.response[:120]}", flush=True)
            record = {
                "id":               idx,
                "conversation":     conv,
                "response":         result.response,
                "tokens_generated": result.tokens_generated,
                "influences":       result.influences,
                "violations":       result.violations,
                "delta_norm":       result.delta_norm,
                "prompt_full_len":  result.prompt_full_len,
                "beta":             beta,
                "beta_decay":       beta_decay,
                "inference_time_s":   round(_elapsed, 4),
                "latency_ms_per_tok": round(_elapsed / max(result.tokens_generated, 1) * 1000, 2),
                "throughput_tok_s":   round(result.tokens_generated / max(_elapsed, 1e-9), 2),
                "est_gflops":         round(_gflops, 2),
                "gflops_per_s":       round(_gflops / max(_elapsed, 1e-9), 2),
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [OOM] skipped", flush=True)
            record = {"id": idx, "response": None, "error": "OOM"}
        except Exception as e:
            import traceback
            print(f"  [ERR] {e}", flush=True)
            traceback.print_exc()
            record = {"id": idx, "response": None, "error": str(e)}

        results.append(record)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} results → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_MULTI_TURN_DIR = Path(__file__).resolve().parent.parent / "data" / "multi-turn"
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "ih-jsd-bias-responses"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file",  type=str, default=None)
    parser.add_argument("--output-dir",  type=str, default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--n",           type=int, default=None)
    parser.add_argument("--max-tokens",  type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--beta",       type=float, default=700.0)
    parser.add_argument("--beta-decay", type=float, default=0.97)
    # Llama (torchtune) model args
    parser.add_argument("--checkpoint-dir",  type=str, default=LocalModel.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-file", type=str, default=LocalModel.DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--tokenizer-path",  type=str, default=LocalModel.DEFAULT_TOKENIZER_PATH)
    # HuggingFace model args (qwen 등)
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "qwen"],
                        help="모델 종류: llama (torchtune) 또는 qwen (HuggingFace)")
    parser.add_argument("--model-path", type=str, default=HFModel.DEFAULT_MODEL_PATH,
                        help="HuggingFace 모델 경로 (--model-type qwen 일 때 사용)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype",  type=str, default="bf16")
    args = parser.parse_args()

    if args.model_type == "qwen":
        model = HFModel(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_seq_len=args.max_seq_len,
            enable_thinking=False,
        )
    else:
        model = LocalModel(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_file=args.checkpoint_file,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            dtype=args.dtype,
            max_seq_len=args.max_seq_len,
        )
    decoder = IHJSDBiasDecoderV5(model)

    if args.input_file:
        input_path = Path(args.input_file)
    else:
        input_path = _DEFAULT_MULTI_TURN_DIR / "both_conflict_default_request.json"

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        tag = f"beta{args.beta}_decay{args.beta_decay}"
        output_path = (
            Path(args.output_dir)
            / f"{args.model_type}_v5_{args.input_file[17:]}_{tag}.json"
        )

    process_file(
        decoder, input_path, output_path,
        max_new_tokens=args.max_tokens,
        beta=args.beta,
        beta_decay=args.beta_decay,
        n=args.n,
        verbose=False,
    )


if __name__ == "__main__":
    main()