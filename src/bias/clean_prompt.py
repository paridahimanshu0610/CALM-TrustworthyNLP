import json
from pathlib import Path
import os

def clean_prompt(text: str) -> str:
    # Remove "Human: \n" from the beginning if present
    if text.startswith("Human: \n"):
        text = text[len("Human: \n"):]

    # Find the last occurrence of "\nAnswer:" and keep text up to and including it
    last_answer_idx = text.rfind("\nAnswer:")
    if last_answer_idx != -1:
        text = text[: last_answer_idx + len("\nAnswer:")]

    # Strip trailing whitespace after truncation
    return text.strip()


# === File paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
input_path = Path("./chatgpt/flare_german_desc/flare_german_desc_write_out_info.json")
output_path = input_path.with_name("flare_german_desc_cleaned.json")

# === Load JSON data ===
with open(input_path, "r") as f:
    data = json.load(f)  # assuming the file is a list of dicts

# === Clean each prompt ===
for entry in data:
    if "prompt_0" in entry:
        entry["prompt_0"] = clean_prompt(entry["prompt_0"])

# === Save cleaned data ===
with open(output_path, "w") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Cleaned prompts saved to: {output_path}")