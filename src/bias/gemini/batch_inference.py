import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import time

def clean_prompt(text: str) -> str:
    if text.startswith("Human: \n"):
        text = text[len("Human: \n"):]
    last_answer_idx = text.rfind("\nAnswer:")
    if last_answer_idx != -1:
        text = text[: last_answer_idx + len("\nAnswer:")]
    return text.strip()

# ---------- Setup ----------
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

MODEL_NAME = "gemini-2.5-flash-lite"

# ---------- File paths ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_cleaned.json")
output_path = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_new_outputs.json")

# ---------- Load JSON ----------
with open(input_path, "r", encoding="utf-8") as f:
    entries = json.load(f)
    entries = entries[0:30]  # limit entries

# ---------- Clean Prompts ----------
for e in entries:
    e["prompt_0"] = clean_prompt(e["prompt_0"])

# ---------- Sequential Inference ----------
results = []
print("Starting sequential inference with 6s delay between requests...")

for i, entry in enumerate(entries, 1):
    success = False
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(entry["prompt_0"])
            model_output = response.text.strip() if response and response.text else ""
            success = True
            break
        except Exception as e:
            print(f"Error on attempt {attempt+1} for entry {i}: {e}")
            time.sleep(2)  # wait before retry

    if not success:
        model_output = ""  # fallback for failed requests

    results.append({
        **entry,
        "logit_0": model_output,
        "model": MODEL_NAME
    })

    print(f"Processed entry {i}/{len(entries)}")
    time.sleep(6)  # <-- ensures <10 requests per minute

# ---------- Save ----------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Finished processing {len(results)} entries. Saved to {output_path}")
