# gpt2_predictor.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio as gr

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Predict next word using GPT-2
def predict_next_word(word1, word2, word3):
    input_text = f"{word1} {word2} {word3}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 1,
            do_sample=False,
            num_return_sequences=1
        )

    generated_text = tokenizer.decode(outputs[0])
    next_word = generated_text[len(input_text):].strip().split(" ")[0]
    return next_word if next_word else "(No word predicted)"

# Gradio Interface
iface = gr.Interface(
    fn=predict_next_word,
    inputs=[
        gr.Textbox(label="First word"),
        gr.Textbox(label="Second word"),
        gr.Textbox(label="Third word"),
    ],
    outputs=gr.Textbox(label="Predicted next word"),
    title="GPT-2 Word Predictor",
    description="Enter three words, and GPT-2 will predict the next one.",
    examples=[["The", "quick", "brown"], ["I", "am", "very"], ["Let", "us", "go"]],
)

if __name__ == "__main__":
    iface.launch()