#!/usr/bin/env python3
"""Augment documents using any OpenAI-compatible LLM endpoint.

Reads document texts from a source directory, evaluates complexity,
and transforms them (simpler/sophisticated) and adds questions.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import math

import yaml

# Try to import textstat, provide a fallback if not available (though it should be)
try:
    import textstat
except ImportError:
    print("ERROR: pip install textstat")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers ──────────────────────────────────────────────────────────


def _read_documents(path):
    """Read newline-separated documents from a text file."""
    texts, doc_ids = [], []
    with open(path, "r", encoding="utf-8") as f:
        doc_id = 0
        current = []
        for line in f:
            current.append(line.rstrip("\n"))
            if line.strip() == "" and len(current) > 1:
                text = "\n".join(current).strip()
                current = []
                if len(text) >= 50:
                    texts.append(text)
                    doc_ids.append(doc_id)
                    doc_id += 1
        if current:
            text = "\n".join(current).strip()
            if len(text) >= 50:
                texts.append(text)
                doc_ids.append(doc_id)
                doc_id += 1
    return texts, doc_ids


def _count_words(text):
    return len(text.split())


# ── prompt engineering ───────────────────────────────────────────────

def build_prompt(doc_text: str, flesch_score: float) -> str:
    """Determine task based on score and build prompt."""
    
    # Calculate number of questions: 1 per 200 words, max 5
    num_words = _count_words(doc_text)
    num_questions = min(5, max(1, math.ceil(num_words / 200)))

    if flesch_score > 80:
        # Easy -> Sophisticated
        task = (
            f"Rewrite the following text to be more sophisticated, academic, and challenging for a "
            f"high-level reader. Then, generate {num_questions} questions that are answered by the "
            f"text. Format your response exactly as follows:\n"
            f"[Sophisticated Text]\n\n"
            f"--- Questions ---\n"
            f"1. [Question 1]\n"
            f"2. [Question 2]\n"
            f"... (up to {num_questions})\n\n"
            f"Original Text: {doc_text}"
        )
    elif flesch_score < 60:
        # Difficult -> Simpler
        task = (
            f"Rewrite the following text to be simpler and easier to understand for a general audience. "
            f"Then, generate {num_questions} questions that are answered by the text. Format your response "
            f"exactly as follows:\n"
            f"[Simpler Text]\n\n"
            f"--- Questions ---\n"
            f"1. [Question 1]\n"
            f"2. [Question 2]\n"
            f"... (up to {num_questions})\n\n"
            f"Original Text: {doc_text}"
        )
    else:
        # Neutral -> Just questions
        task = (
            f"Generate {num_questions} questions that are answered by the following text. "
            f"Format your response exactly as follows:\n"
            f"[Original Text]\n\n"
            f"--- Questions ---\n"
            f"1. [Question 1]\n"
            f"2. [Question 2]\n"
            f"... (up to {num_questions})\n\n"
            f"Original Text: {doc_text}"
        )
    
    return task


# ── core logic ───────────────────────────────────────────────────────


def build_client(config):
    """Establish an OpenAI client for the configured endpoint."""
    import openai
    from openai import OpenAI

    endpoint = os.environ.get(
        "OPENAI_API_BASE",
        config.get("endpoint", "http://localhost:11434/v1"),
    )
    api_key = os.environ.get(
        "OPENAI_API_KEY",
        config.get("api_key", "ollama"),
    )
    return OpenAI(base_url=endpoint, api_key=api_key)


def generate_batch(client, model, prompts, max_tokens, retry_delay):
    """Call the OpenAI chat completions endpoint for a batch of prompts."""
    results = []
    for prompt in prompts:
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites text and generates questions."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content
            results.append(("OK", text if text else None))
        except Exception as e:
            print(f"    API error: {e}")
            results.append(("ERR", None))
            time.sleep(retry_delay)
    return results


def process_file(
    src_file, prefix, input_dir, output_subdir, config, client, model,
    max_tokens, batch_size, retry_delay
):
    """Process one source file: read docs, generate, write."""
    docs, doc_ids = _read_documents(str(src_file))
    print(f"  {src_file.name}: {len(docs)} documents")

    # Find already augmented docs
    already = set()
    for meta_f in output_subdir.glob("*.jsonl.meta"):
        try:
            entry = json.loads(meta_f.read_text().strip())
            already.add(entry["doc_id"])
        except Exception:
            already.add(meta_f.stem)
    print(f"  Already augmented: {len(already)}")

    pending = [(did, doc) for did, doc in zip(doc_ids, docs) if did not in already]
    if not pending:
        print("  All done, skipping.")
        return 0, 0

    n_aug = 0
    n_err = 0
    t0 = time.time()
    n_batch = 0

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        
        # Prepare prompts with complexity info
        prompts = []
        for _, doc in batch:
            score = textstat.flesch_reading_ease(doc)
            prompts.append(build_prompt(doc, score))
            
        ids = [did for did, _ in batch]

        responses = generate_batch(client, model, prompts, max_tokens, retry_delay)

        for did, doc, (status, response) in zip(ids, batch, responses):
            if status != "OK" or not response:
                n_err += 1
                continue

            path = output_subdir / f"{prefix}_{did:06d}.txt"
            path.write_text(response, encoding="utf-8")

            meta_path = output_subdir / f"{prefix}_{did:06d}.jsonl.meta"
            meta_path.write_text(
                json.dumps({"doc_id": did, "source": prefix,
                            "provider": "openai_compatible", "model": model})
                + "\n", encoding="utf-8")

            n_aug += 1

        n_batch += 1
        dt = time.time() - t0
        print(f"  Batch {n_batch}: {n_aug} augmented | {n_err} err | {dt:.0f}s")

    return n_aug, n_err


def main():
    parser = argparse.ArgumentParser(
        description="Augment documents with an OpenAI-compatible LLM")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with deduped source files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for augmented output")
    parser.add_argument("--config", required=True,
                        help="LLM config file (yaml)")
    parser.add_argument("--model", type=str,
                        help="Override model name from config")
    parser.add_argument("--max-tokens", type=int,
                        help="Override max tokens from config")
    parser.add_argument("--batch-size", type=int,
                        help="Override batch size from config")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load config
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    model = os.environ.get("OPENAI_MODEL", args.model or base_cfg.get("model", "llama3.1"))
    max_tokens = args.max_tokens or base_cfg.get("max_tokens", 2048)
    batch_size = args.batch_size or base_cfg.get("batch_size", 10)
    retry_delay = base_cfg.get("retry_delay", 3.0)

    # Check deps
    try:
        import openai
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    client = build_client(base_cfg)
    print(f"  Endpoint: {client.base_url}")
    print(f"  Model: {model}  Max tokens: {max_tokens}  Batch: {batch_size}")

    # Process each source file in the input dir
    # We exclude 'code_clean.txt' as it is code, not text.
    # We look for files in the input_dir.
    
    source_files_map = {
        "fineweb_edu_hq.txt": "fineweb_edu",
        "wikipedia_clean.txt": "wikipedia",
        "stackexchange_clean.txt": "stackexchange",
        "arxiv_clean.txt": "arxiv",
    }

    total_aug, total_err = 0, 0

    for filename, prefix in source_files_map.items():
        src = input_dir / filename
        if not src.exists():
            print(f"[skip] {filename} not found in {input_dir}")
            continue

        out_sub = output_dir / prefix
        out_sub.mkdir(parents=True, exist_ok=True)

        n_aug, n_err = process_file(
            src, prefix, input_dir, out_sub, base_cfg, client, model,
            max_tokens, batch_size, retry_delay
        )
        total_aug += n_aug
        total_err += n_err

    print(f"\n  Total: {total_aug} augmented, {total_err} failed")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
