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
input_path = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_debias_input.json")
output_path = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_debias_llm_output.json")

# ---------- Load JSON ----------
with open(input_path, "r", encoding="utf-8") as f:
    entries = json.load(f)
    entries = entries[0:50]  # limit entries

# ---------- Clean Prompts ----------
# for e in entries:
#     e["prompt_0"] = clean_prompt(e["query"])

# ---------- Sequential Inference ----------
results = []
print("Starting sequential inference with 6s delay between requests...")

def transform_dict(data: dict) -> dict:
    """
    Transforms the input dictionary into the required output format.
    """
    doc_id = data.get("id")
    query = data.get("query", "")
    predicted_answer = data.get("predicted_answer", "").strip().lower()
    truth = data.get("answer", "").strip().lower()

    # Compute accuracy
    acc = "1.0" if predicted_answer == truth else "0.0"

    # Compute missing (1 if predicted answer is not 'good' or 'bad')
    missing = "0" if predicted_answer in ["good", "bad"] else "1"

    # F1, macro_f1, and MCC (all same tuple format)
    metric_tuple = (predicted_answer, truth)

    transformed = {
        "doc_id": doc_id,
        "prompt_0": query,
        "logit_0": predicted_answer,
        "truth": truth,
        "acc": acc,
        "missing": missing,
        "f1": str(metric_tuple),
        "macro_f1": str(metric_tuple),
        "mcc": str(metric_tuple)
    }

    return transformed

for i, entry in enumerate(entries, 1):
    success = False
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(entry["query"])
            model_output = response.text.strip().lower() if response and response.text else ""
            success = True
            break
        except Exception as e:
            print(f"Error on attempt {attempt+1} for entry {i}: {e}")
            time.sleep(61)  # wait before retry

    if not success:
        model_output = ""  # fallback for failed requests

    temp = {
        **entry,
        "predicted_answer": model_output,
        "model": MODEL_NAME
    }
    temp = transform_dict(temp)
    results.append(temp)

    print(f"Processed entry {i}/{len(entries)}")
    time.sleep(10)  # <-- ensures <6 requests per minute

# ---------- Save ----------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Finished processing {len(results)} entries. Saved to {output_path}")
