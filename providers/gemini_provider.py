from __future__ import annotations

from google import genai
from google.genai import types
import os
from typing import Optional

from .base import BaseProvider

class GeminiProvider(BaseProvider):
  def __init__(self, model, max_input_tokens, tpm_budget = 420000, retries = 5, api_key: Optional[str] = None, temperature: float = 0.0) -> None:
    super().__init__(model, max_input_tokens, tpm_budget, retries)
    self.temperature = temperature
    if api_key is None:
      raise ValueError('Argument api_key not found.')
    
    if api_key:
      self.client = genai.Client(api_key=api_key)
    else:
      self.client = genai.Client()
  
  def _generate_once(self, prompt: str, max_output_tokens: Optional[int] = None):
    config_kwargs = {
      'max_output_tokens': max_output_tokens,
      'temperature': self.temperature,
      'thinking_config': types.ThinkingConfig(thinking_budget=128),
    }

    response = self.client.models.generate_content(model=self.model, contents=prompt, config=types.GenerateContentConfig(**config_kwargs))
    return getattr(response, 'text', '') or ''