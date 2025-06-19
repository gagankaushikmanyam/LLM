# trigram_predictor.py

import nltk
nltk.download('brown')
from nltk.corpus import brown
from collections import defaultdict, Counter
import gradio as gr

# Build Trigram Language Model
trigram_model = defaultdict(Counter)

for sentence in brown.sents():
    sentence = ['<s>'] + [word.lower() for word in sentence] + ['</s>']
    for i in range(len(sentence) - 3):
        w1, w2, w3 = sentence[i], sentence[i+1], sentence[i+2]
        next_word = sentence[i+3]
        trigram_model[(w1, w2, w3)][next_word] += 1

# Prediction function
def predict_next(w1, w2, w3):
    key = (w1.lower(), w2.lower(), w3.lower())
    if key in trigram_model:
        return trigram_model[key].most_common(1)[0][0]
    else:
        return "(unknown)"

# Gradio Interface
demo = gr.Interface(
    fn=predict_next,
    inputs=[
        gr.Textbox(label="Word 1"),
        gr.Textbox(label="Word 2"),
        gr.Textbox(label="Word 3"),
    ],
    outputs=gr.Textbox(label="Predicted 4th Word"),
    title="Next Word Predictor (Trigram Model)",
    description="Enter 3 English words, and this trigram-based language model predicts the most likely 4th word."
)

if __name__ == "__main__":
    demo.launch()