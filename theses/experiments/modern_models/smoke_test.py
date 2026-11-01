"""
Smoke test to verify that all configured modern models are reachable and functional.
Checks each model in MODEL_CONFIGS with a simple request.
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import List, Tuple

from src.infra.llm.client import LLMClient
from src.infra.llm.models import Models, ReasoningEffort
from src.infra.llm.message import Message, MessageRole
from .model_config import (
    MODEL_CONFIGS, 
    get_model_reasoning_effort, 
    get_model_extra_body, 
    get_config_description
)

async def test_model_config(client: LLMClient, model: Models, with_reasoning: bool) -> Tuple[str, bool, str]:
    """Test a single model configuration."""
    desc = get_config_description(model, with_reasoning)
    effort = get_model_reasoning_effort(model, with_reasoning)
    extra_body = get_model_extra_body(model, disable_cache=True, with_reasoning=with_reasoning)
    
    print(f"Testing: {desc}...")
    print(f"  - Model: {model.model_name}")
    print(f"  - Reasoning: {effort.value if effort else 'None'}")
    
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Say exactly the word 'PONG' and nothing else.")
    ]
    
    # For Claude reasoning mode, temperature must be exactly 1.0
    # Also max_tokens must be greater than thinking.budget_tokens
    # LiteLLM maps reasoning_effort=high to a budget. Let's use a safe max_tokens.
    temp = 1.0 if with_reasoning and (model == Models.CLAUDE_SONNET_4_5 or model == Models.GEMINI_3_FLASH) else 0.0
    max_t = 16000 if with_reasoning else 100
    
    try:
        # We use a short timeout for smoke tests
        response = await asyncio.wait_for(
            client.generate_async(
                model=model,
                messages=messages,
                temperature=temp,
                reasoning_effort=effort,
                extra_body=extra_body,
                max_tokens=max_t
            ),
            timeout=90.0 # Increased timeout for reasoning
        )
        
        content = str(response.content).strip()
        success = "PONG" in content.upper()
        return desc, success, content
    except Exception as e:
        return desc, False, str(e)

async def main():
    load_dotenv()
    
    # Check for LiteLLM proxy
    # The client defaults to OpenAI() which reads OPENAI_API_BASE from env
    # If not set, it points to direct OpenAI API.
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    print(f"LiteLLM API Base: {api_base}")
    
    client = LLMClient()
    
    print(f"\n{'='*60}")
    print(f"Modern Models Smoke Test - {len(MODEL_CONFIGS)} configurations")
    print(f"{'='*60}\n")
    
    results = []
    for model, with_reasoning in MODEL_CONFIGS:
        result = await test_model_config(client, model, with_reasoning)
        results.append(result)
        
        desc, success, output = result
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"Result: {status}")
        if not success:
            print(f"Error/Output: {output}")
        print("-" * 40)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for desc, success, output in results:
        status = "✅" if success else "❌"
        print(f"{status} {desc}")
        
    print(f"\nPassed: {passed}/{len(results)}")
    
    if failed > 0:
        print("\nSome models failed. Please check your API keys and LiteLLM proxy status.")
        exit(1)
    else:
        print("\nAll models are operational! 🚀")

if __name__ == "__main__":
    asyncio.run(main())
