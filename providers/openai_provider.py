from __future__ import annotations

from openai import OpenAI
import os
from typing import Optional

from .base import BaseProvider

class OpenAIProvider(BaseProvider):
  def __init__(self, model, max_input_tokens, tpm_budget = 420000, retries = 5, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
    super().__init__(model, max_input_tokens, tpm_budget, retries)
    if api_key is None:
      raise ValueError('Argument api_key not found.')

    if base_url:
      self.client = OpenAI(api_key=api_key, base_url=base_url)
    else:
      self.client = OpenAI(api_key=api_key)
  
  def _generate_once(self, prompt: str, max_output_tokens: Optional[int] = None):
    response = self.client.responses.create(model=self.model, input=prompt, reasoning={"effort": "minimal"}, max_output_tokens=max_output_tokens)
    return response.output_text or ""