#!/usr/bin/env python3
"""Interactive multi-turn chat with Llama 3.1 70B via TurboQuant.

Streams tokens in real-time. Full conversation history is sent each turn.
Uses the fused int4 SDPA kernel for decode — the same kernel that enables
128K context on 64GB Apple Silicon.

Usage:
    python3.12 engine/chat_repl.py [model_dir]

Commands:
    quit  — exit
    clear — reset conversation memory
"""
import os, sys, subprocess, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build", "chat_engine")

# Default model directory
DEFAULT_MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-70B-Instruct-4bit/"
    "snapshots/7772c93cf077b642f5503dd8d763a4176d7d406c/"
)

# ANSI codes
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"
CYAN = "\033[36m"

# Load tokenizer
def load_tokenizer(model_dir):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir)
    except ImportError:
        print("Error: transformers not installed. Run: pip3.12 install transformers")
        sys.exit(1)


SYSTEM_PROMPT = """\
You are a helpful AI assistant. You are Llama 3.1 70B running locally on \
Apple Silicon via TurboQuant — a custom C++ inference engine with fused int4 \
SDPA Metal kernels that compress the KV cache 3.2x, enabling 128K context \
on 64GB hardware.

Be direct, concise, and helpful. Answer the user's question without preamble."""


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR

    if not os.path.exists(BINARY):
        print(f"Error: {BINARY} not found.")
        print(f"Run: cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8")
        sys.exit(1)

    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}")
        print(f"Run: python3.12 -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download('mlx-community/Meta-Llama-3.1-70B-Instruct-4bit')\"")
        sys.exit(1)

    print(f"\n{BOLD}{ORANGE}Llama 3.1 70B{RESET} — TurboQuant Fused int4 SDPA")
    print(f"{DIM}128K context on 64GB  |  int4 KV cache (3.2x compression){RESET}")
    print(f"{DIM}Loading model (39 GB)...{RESET}", end="", flush=True)

    tok = load_tokenizer(model_dir)

    # Start the C++ engine
    proc = subprocess.Popen(
        [BINARY, model_dir, "256"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=ROOT,
    )

    # Wait for model to load (monitor stderr for status)
    def read_stderr():
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            msg = line.decode("utf-8", errors="replace").strip()
            if msg == "MODEL_READY":
                return True
            elif msg.startswith("LOADING"):
                pass
            elif msg.startswith("[loader]"):
                # Show loading progress
                sys.stderr.write(f"\r{DIM}{msg}{RESET}    ")
                sys.stderr.flush()
        return False

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    # Wait for READY on stdout
    while True:
        line = proc.stdout.readline().decode("utf-8").strip()
        if line == "READY":
            break
        if not line and proc.poll() is not None:
            print(f"\n{BOLD}Error:{RESET} Engine failed to start.")
            remaining = proc.stderr.read().decode("utf-8", errors="replace")
            if remaining:
                print(remaining[:500])
            sys.exit(1)

    print(f"\r{DIM}Model loaded.{RESET}                              ")
    print(f"Type {BOLD}quit{RESET} to exit, {BOLD}clear{RESET} to reset memory\n")

    history = []

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            print(f"{DIM}[Memory cleared]{RESET}\n")
            continue

        history.append({"role": "user", "content": user_input})

        # Build the full conversation with chat template
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tok.encode(prompt, add_special_tokens=False)

        n_prompt = len(token_ids)
        if n_prompt > 2000:
            print(f"  {DIM}[context: {n_prompt} tokens]{RESET}")

        # Send to engine
        ids_str = " ".join(str(t) for t in token_ids)
        proc.stdin.write(f"TOKEN_IDS: {ids_str}\n".encode("utf-8"))
        proc.stdin.flush()

        # Show spinner during prefill
        spinning = True
        def spinner():
            frames = [".", "..", "..."]
            i = 0
            while spinning:
                sys.stdout.write(f"\r  {DIM}thinking{frames[i % len(frames)]}{RESET}   ")
                sys.stdout.flush()
                i += 1
                time.sleep(0.3)
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()

        spin_thread = threading.Thread(target=spinner, daemon=True)
        spin_thread.start()

        # Read streaming tokens
        response_tokens = []
        first_token = True

        while True:
            line = proc.stdout.readline().decode("utf-8").strip()
            if not line:
                if proc.poll() is not None:
                    spinning = False
                    print(f"\n{BOLD}Error:{RESET} Engine crashed.")
                    sys.exit(1)
                continue

            if line.startswith("TOKEN "):
                token_id = int(line.split()[1])
                response_tokens.append(token_id)

                if first_token:
                    spinning = False
                    spin_thread.join(timeout=1)
                    sys.stdout.write(f"{BOLD}{ORANGE}Llama:{RESET} ")
                    first_token = False

                # Decode incrementally and print new text
                decoded = tok.decode(response_tokens)
                # Print only new characters since last decode
                prev_decoded = tok.decode(response_tokens[:-1]) if len(response_tokens) > 1 else ""
                new_text = decoded[len(prev_decoded):]
                sys.stdout.write(new_text)
                sys.stdout.flush()

            elif line == "DONE":
                break

        spinning = False
        print()

        # Read decode stats from stderr
        time.sleep(0.1)
        try:
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                msg = line.decode("utf-8", errors="replace").strip()
                if msg.startswith("DECODE_DONE"):
                    parts = msg.split()
                    n_tok = parts[1]
                    tok_s = parts[-2] if len(parts) > 3 else "?"
                    print(f"  {DIM}[{n_tok} tokens, {tok_s} tok/s]{RESET}")
                    break
                elif msg.startswith("PREFILL_DONE"):
                    pass  # Already handled by spinner
        except Exception:
            pass

        print()

        # Add assistant response to history
        if response_tokens:
            response_text = tok.decode(response_tokens)
            history.append({"role": "assistant", "content": response_text})

    # Cleanup
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
