from dotenv import load_dotenv
import openai, json, os

# === File paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
print(open_ai_api_key)
openai.api_key = open_ai_api_key 

input_file = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_cleaned.json")
output_file = os.path.join(current_dir, 'flare_german_desc', "flare_german_desc_new_outputs.json")


with open(input_file, "r") as f:
    data = json.load(f)

results = []
for entry in data[0:2]:
    prompt = entry["prompt_0"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    output = response["choices"][0]["message"]["content"].strip()

    new_entry = {
        **entry,
        "logit_0": output,  # model output
        "model": "gpt-3.5-turbo"
    }
    results.append(new_entry)

with open(output_file, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Model responses saved to: {output_file}")