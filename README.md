# Creative Math Problem Solver

## Project Overview
The **Creative Math Problem Solver** is a generative AI project designed to solve math problems written entirely in emoji. The goal is to fine-tune a transformer-based model to understand and compute emoji-based equations, pushing the boundaries of generative AI with creativity and mathematical reasoning.

## Features
- **Fine-tune a large language model** to solve emoji-based math problems.
- **Create and preprocess a dataset** of 30 emoji math problems.
- **Train the model with LoRA (Low-Rank Adaptation)** to improve efficiency.
- **Use 4-bit quantization** to optimize memory usage.
- **Test the model** on new emoji-based equations.

## Dataset
The dataset consists of emoji-based mathematical expressions with solutions. Example:
```
🍏 + 🍏 + 🍏 = 12 → 🍏 = 4
🚗 + 🚗 + 🚗 + 🚗 = 20 → 🚗 = 5
```
### Preprocessing
- The dataset is stored in a CSV file with columns `problem` and `solution`.
- The text is formatted as:
  ```
  Problem Statement: <emoji-equation>\nSolution: <solution>
  ```
- The dataset is tokenized using `AutoTokenizer` from Hugging Face Transformers.

## Model Architecture
- **Base Model:** `deepseek-ai/deepseek-math-7b-base` (A transformer model trained for mathematical reasoning).
- **LoRA Fine-Tuning:**
  - Rank (r): 20
  - LoRA Alpha: 40
  - Target modules: `q_proj`, `v_proj`
  - LoRA Dropout: 0.05
- **Quantization:** 4-bit quantization with `BitsAndBytesConfig` for optimized performance.

## Training Process
1. Load and preprocess the dataset.
2. Fine-tune the model using LoRA.
3. Use AdamW optimizer with a learning rate of `2e-5`.
4. Train for `3 epochs` with batch size `1`.
5. Save the fine-tuned model to `finetuned_deepseek_math`.

## Testing & Results
The fine-tuned model is evaluated using new emoji-based problems:
```
🚗 + 🚗 + 🚗 + 🚗 = 20 → 🚗 = ?
🎈 + 🎈 + 🎈 = 15 → 🎈 = ?
🐶 + 🐶 = 12 → 🐶 = ?
```
The model successfully generates correct solutions for the emoji equations.

## Installation & Setup
### Dependencies
Install the required libraries:
```bash
pip install transformers accelerate peft datasets bitsandbytes torch
```
### Running the Model
#### Training
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-base")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-7b-base")

# Fine-tune the model using LoRA
model = get_peft_model(model, lora_config)
```
#### Inference
```python
input_text = "🚗 + 🚗 + 🚗 + 🚗 = 20 → 🚗 = ?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Future Improvements
- Expand the dataset with more complex emoji-based math problems.
- Fine-tune on larger models for better accuracy.
- Deploy as an interactive web app using Streamlit.
- Implement multi-step problem-solving capabilities.

---
This project demonstrates how generative AI can be creatively applied to solve unconventional problems while maintaining mathematical reasoning capabilities.

## Results
![Emoji Math](https://github.com/Muradhameed921/Sudoku-Puzzle-Solver/blob/main/O1.jpg)
