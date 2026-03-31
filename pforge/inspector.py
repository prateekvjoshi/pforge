"""
inspector.py — standalone model introspection script (run as subprocess).

Performs a logit lens analysis: for a given prompt, applies the model's
final layer norm + unembedding matrix to the hidden state at every
transformer layer.  This reveals what the model "would predict" at each
depth — showing how the answer crystallises from noise to certainty as
information propagates through the network.

Run via server POST /logit_lens (never call directly in production).
The server stops vLLM first so this script has full VRAM access.

Output: writes a JSON file to --output_path, then exits.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inspector")


def run_logit_lens(
    model_name: str,
    prompt: str,
    top_k: int,
    output_path: Path,
    lora_path: str = "",
) -> None:
    logger.info("Loading model %s for logit lens analysis", model_name)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Optionally load a LoRA adapter so users can compare base vs fine-tuned
    if lora_path:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s", lora_path)
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # merge for clean hidden state access

    model.eval()
    logger.info("Model loaded. Running forward pass with output_hidden_states=True")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    # Decode input tokens for display
    input_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states: tuple of n_layers+1 tensors, each (1, seq_len, hidden_dim)
    # Index 0 = token embeddings; index k = output of transformer layer k-1
    hidden_states = outputs.hidden_states

    # Get the model's final layer norm and unembedding matrix
    # Works for Qwen3 and most decoder-only models (LLaMA, Mistral, etc.)
    base = model.model if hasattr(model, "model") else model
    final_norm = base.norm
    lm_head    = model.lm_head

    logger.info(
        "Running logit lens across %d hidden states (1 embedding + %d layers)",
        len(hidden_states), len(hidden_states) - 1,
    )

    layers_output = []
    answer_token = None
    answer_first_layer = None

    for idx, hs in enumerate(hidden_states):
        # Apply final norm + unembedding to get logits at this depth
        normed = final_norm(hs.float())          # (1, seq_len, hidden_dim)
        logits = lm_head(normed.to(model.dtype)) # (1, seq_len, vocab_size)

        # Focus on the last token position — predicting what comes next
        last_logits = logits[0, -1, :]           # (vocab_size,)
        probs       = torch.softmax(last_logits.float(), dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)

        predictions = []
        for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
            token_str = tokenizer.decode([tid])
            predictions.append({
                "token":    token_str,
                "token_id": tid,
                "prob":     round(prob, 4),
            })

        top1_token = predictions[0]["token"] if predictions else ""

        # Track when the final answer first appears as top-1 prediction
        if answer_token is None and idx > 0:
            answer_token = top1_token
        if (answer_first_layer is None
                and top1_token == answer_token
                and idx > 0):
            answer_first_layer = idx

        label = "embedding" if idx == 0 else f"layer_{idx}"
        layers_output.append({
            "layer":           idx,
            "label":           label,
            "top_predictions": predictions,
        })

    # Final answer is top-1 at the last layer
    final_answer = layers_output[-1]["top_predictions"][0]["token"] if layers_output else ""

    # Recompute answer_first_layer relative to final answer
    answer_first_layer = None
    for entry in layers_output[1:]:  # skip embedding layer
        if entry["top_predictions"][0]["token"] == final_answer:
            answer_first_layer = entry["layer"]
            break

    result = {
        "prompt":                    prompt,
        "model":                     model_name,
        "lora_path":                 lora_path or None,
        "input_tokens":              input_tokens,
        "num_layers":                len(hidden_states) - 1,
        "layers":                    layers_output,
        "final_answer":              final_answer,
        "answer_first_appears_at_layer": answer_first_layer,
        "timestamp":                 datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Results written to %s", output_path)

    # Cleanup
    del model, outputs, hidden_states
    torch.cuda.empty_cache()
    import gc; gc.collect()
    logger.info("CUDA memory after cleanup: %.2f GB", torch.cuda.memory_allocated() / 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   required=True)
    parser.add_argument("--prompt_file",  required=True,
                        help="Path to a JSON file containing {\"prompt\": \"...\"}. "
                             "Using a file instead of a CLI arg avoids shell injection risks.")
    parser.add_argument("--output_path",  required=True)
    parser.add_argument("--top_k",        type=int, default=5)
    parser.add_argument("--lora_path",    default="")
    args = parser.parse_args()

    # Read prompt from file (never from a raw CLI arg — avoids injection)
    prompt_data = json.loads(Path(args.prompt_file).read_text())
    prompt = prompt_data["prompt"]

    try:
        run_logit_lens(
            model_name=  args.model_name,
            prompt=      prompt,
            top_k=       args.top_k,
            output_path= Path(args.output_path),
            lora_path=   args.lora_path,
        )
        sys.exit(0)
    except Exception:
        import traceback
        logger.error("Logit lens failed:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
