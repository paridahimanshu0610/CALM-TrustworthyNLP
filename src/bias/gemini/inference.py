import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

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

# ---------- Setup ----------
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

MODEL_NAME = "gemini-2.5-flash"  # or "gemini-1.5-pro"

# === File paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

input_path = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_cleaned.json")
output_path = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_new_outputs.json")

with open(input_path, "r", encoding="utf-8") as f:
    entries = json.load(f)

# ---------- Initialize Model ----------
model = genai.GenerativeModel(MODEL_NAME)

# ---------- Process ----------
updated_entries = []
for entry in entries[0:10]:
    prompt = entry["prompt_0"] #clean_prompt(entry["prompt_0"])

    response = model.generate_content(prompt)
    model_output = response.text.strip()

    new_entry = {
        **entry,
        "prompt_0": prompt,
        "logit_0": model_output,   # replace with model prediction
        "model": MODEL_NAME
    }
    updated_entries.append(new_entry)

# ---------- Save ----------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(updated_entries, f, indent=4, ensure_ascii=False)

print(f"Processed {len(updated_entries)} entries and saved to {output_path}")