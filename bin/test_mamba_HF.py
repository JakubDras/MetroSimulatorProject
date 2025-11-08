import torch
from transformers import MambaForCausalLM, AutoTokenizer

# Upewnij się, że używasz GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {device}")

# Wybierz model Mamba z repozytorium Hugging Face
model_name = "state-spaces/mamba-2.8b-slimpj"

# Załaduj model i tokenizer
# torch_dtype=torch.bfloat16 jest zalecane dla nowszych kart (seria 30xx i nowsze)
model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model Mamba został pomyślnie załadowany!")

# Przykład generowania tekstu
prompt = "Mamba to nowa architektura w dziedzinie sztucznej inteligencji, która"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(**inputs, max_length=50, eos_token_id=tokenizer.eos_token_id)

# Zdekoduj i wyświetl wynik
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nWygenerowany tekst:")
print(generated_text)