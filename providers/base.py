from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import tiktoken
import time
from typing import Optional

class BaseProvider(ABC):
  """
  Common logic shared by different API providers
  - truncate in the middle
  - rough TPM throttling
  - retry / backoff warpper
  """

  def __init__(self, model: str, max_input_tokens: int, tpm_budget: int = 420000, retries: int = 5,) -> None:
    self.model = model
    self.max_input_tokens = max_input_tokens
    self.tpm_budget = tpm_budget
    self.retries = retries
    self.request_log = deque()
  
  def _encoding(self):
    """
    Get tokenizer from the provider
    """
    try:
      return tiktoken.encoding_for_model(self.model)
    except Exception:
      return tiktoken.get_encoding('o200k_base')
    
  def estimate_tokens(self, text: str) -> int:
    enc = self._encoding()
    return len(enc.encode(text, disallowed_special=()))
  
  def truncate_middle(self, text: str):
    enc = self._encoding()
    text = enc.encode(text, disallowed_special=())
    if len(text) > self.max_input_tokens:
      text = text[:self.max_input_tokens // 2] + text[-self.max_input_tokens // 2:]
    return enc.decode(text)
  
  def _get_used_token_budget(self):
    now = time.time()
    # clean request log
    while self.request_log and now - self.request_log[0][0] > 60:
      self.request_log.popleft()
    return sum(x[1] for x in self.request_log)
  
  def throttle(self, prompt: str, max_output_tokens: int) -> None:
    """
    Per-process TPM throttle
    Only for n_proc=1
    """
    used = self._get_used_token_budget()
    need = self.estimate_tokens(prompt) + max_output_tokens
    while self.request_log and used + need > self.tpm_budget:
      now = time.time()
      wait = 60 - (now - self.request_log[0][0]) + 1
      time.sleep(max(1, wait))
      used = self._get_used_token_budget()
  
  def is_rate_limit_error(self, exc: Exception) -> bool:
    s = str(exc).lower()
    return ('rate limit' in s or 'too many requests' in s or '429' in s or 'resource exhausted' in s)
  
  def generate(self, prompt: str, max_output_tokens: int) -> str:
    """
    truncate middle, throttle, retry
    """
    prompt = self.truncate_middle(prompt)

    last_err = None
    for attemp in range(self.retries):
      try:
        self.throttle(prompt, max_output_tokens)
        text = self._generate_once(prompt, max_output_tokens)
        self.request_log.append((time.time(), self.estimate_tokens(prompt) + max_output_tokens))
        return (text or "").strip()
      except Exception as e:
        last_err = e
        if self.is_rate_limit_error(e):
          wait = min(10, (2 ** attemp))
          time.sleep(wait)
          continue
        time.sleep(1)
    raise RuntimeError(f'Provider call failed after {self.retries} retries: {last_err}')
  
  @abstractmethod
  def _generate_once(self, prompt: str, max_output_tokens: int) -> str:
    raise NotImplementedError