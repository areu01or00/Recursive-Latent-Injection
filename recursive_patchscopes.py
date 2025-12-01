"""
Recursive Patchscope Experiment
- Run normal generation
- Run "recursive" generation: extract late layer, inject into early layer, generate again

Using Qwen3-0.6B with proper chat template
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_NAME = "Qwen/Qwen3-0.6B"
EXTRACT_LAYER = 24  # Late layer to extract from (model has 28 layers)
INJECT_LAYER = 3    # Early layer to inject into
MAX_NEW_TOKENS = 512


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Model loaded. Layers: {num_layers}")
    print(f"Device: {model.device}")
    return model, tokenizer


def prepare_input(tokenizer, prompt, enable_thinking=False):
    """Prepare input using Qwen3 chat template."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    return tokenizer([text], return_tensors="pt")


def parse_output(tokenizer, output_ids):
    """Parse output, separating thinking content from actual content."""
    output_list = output_ids.tolist()

    # Try to find </think> token (151668)
    try:
        index = len(output_list) - output_list[::-1].index(151668)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output_list[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_list[index:], skip_special_tokens=True).strip()

    return thinking, content


def normal_generate(model, tokenizer, prompt, enable_thinking=False):
    """Standard generation without any patching."""
    model_inputs = prepare_input(tokenizer, prompt, enable_thinking).to(model.device)
    input_len = len(model_inputs.input_ids[0])

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
        )

    output_ids = generated_ids[0][input_len:]
    thinking, content = parse_output(tokenizer, output_ids)

    return thinking, content


def recursive_generate(model, tokenizer, prompt, num_iterations=1, enable_thinking=False):
    """
    Recursive generation:
    1. Run forward pass, extract hidden state at EXTRACT_LAYER
    2. Run another forward pass, inject that hidden state at INJECT_LAYER
    3. Generate from the modified state
    """
    model_inputs = prepare_input(tokenizer, prompt, enable_thinking).to(model.device)
    input_ids = model_inputs.input_ids
    input_len = len(input_ids[0])

    # Storage for extracted hidden state
    extracted_hs = None

    def extract_hook(module, input, output):
        nonlocal extracted_hs
        # output[0] shape: (batch, seq_len, hidden_dim)
        extracted_hs = output[0].clone()

    def inject_hook(module, input, output):
        nonlocal extracted_hs
        # Replace output with our extracted hidden state
        # Only inject during initial prompt processing (full sequence), not during
        # token-by-token generation (where seq_len=1 due to KV caching)
        if extracted_hs is not None:
            out_tensor = output[0]
            # Check if this is full sequence (initial) or single token (generation step)
            if out_tensor.dim() == 3 and out_tensor.shape[1] == extracted_hs.shape[1]:
                # Full sequence - inject
                new_output = out_tensor.clone()
                new_output[:, :, :] = extracted_hs
                return (new_output,) + output[1:]
        return output

    # Iterate: extract -> inject -> extract -> inject ...
    for _ in range(num_iterations):
        extracted_hs = None

        # Step 1: Run forward pass and extract from late layer
        handle_extract = model.model.layers[EXTRACT_LAYER].register_forward_hook(extract_hook)

        with torch.no_grad():
            _ = model(input_ids)

        handle_extract.remove()

    # Final generation with injection
    handle_inject = model.model.layers[INJECT_LAYER].register_forward_hook(inject_hook)

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
        )

    handle_inject.remove()

    output_ids = generated_ids[0][input_len:]
    thinking, content = parse_output(tokenizer, output_ids)

    return thinking, content


def main():
    model, tokenizer = load_model()

    print("\n" + "="*60)
    print("RECURSIVE PATCHSCOPE EXPERIMENT")
    print(f"Extract from layer {EXTRACT_LAYER}, inject into layer {INJECT_LAYER}")
    print("="*60)
    print("\nEnter prompts to compare normal vs recursive generation.")
    print("Type 'quit' to exit, 'think' to toggle thinking mode.\n")

    enable_thinking = False

    while True:
        prompt = input(f"\n[thinking={'ON' if enable_thinking else 'OFF'}] Prompt: ").strip()

        if prompt.lower() == 'quit':
            break
        if prompt.lower() == 'think':
            enable_thinking = not enable_thinking
            print(f"Thinking mode: {'ON' if enable_thinking else 'OFF'}")
            continue
        if not prompt:
            continue

        print("\n" + "-"*40)
        print("NORMAL OUTPUT:")
        print("-"*40)
        thinking, content = normal_generate(model, tokenizer, prompt, enable_thinking)
        if thinking:
            print(f"[Thinking]: {thinking[:200]}..." if len(thinking) > 200 else f"[Thinking]: {thinking}")
        print(f"[Response]: {content}")

        print("\n" + "-"*40)
        print("RECURSIVE OUTPUT (1 iteration):")
        print("-"*40)
        thinking, content = recursive_generate(model, tokenizer, prompt, num_iterations=1, enable_thinking=enable_thinking)
        if thinking:
            print(f"[Thinking]: {thinking[:200]}..." if len(thinking) > 200 else f"[Thinking]: {thinking}")
        print(f"[Response]: {content}")

        print("\n" + "-"*40)
        print("RECURSIVE OUTPUT (3 iterations):")
        print("-"*40)
        thinking, content = recursive_generate(model, tokenizer, prompt, num_iterations=3, enable_thinking=enable_thinking)
        if thinking:
            print(f"[Thinking]: {thinking[:200]}..." if len(thinking) > 200 else f"[Thinking]: {thinking}")
        print(f"[Response]: {content}")


if __name__ == "__main__":
    main()
