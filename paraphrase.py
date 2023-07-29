import os
import nltk
from transformers import pipeline

# Step 1: Install required libraries
# pip install nltk transformers torch

# Step 2: Import Libraries
import nltk
from transformers import pipeline

# Step 3: Load Pre-trained Models
paraphrase_model = pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")

# Step 4: Read Input from File
file_path = 'assignment.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Step 5: Preprocess Text (Optional)
# Preprocessing can be performed if needed.

# Step 6: Tokenization (Optional)
nltk.download('punkt')  # Download the Punkt tokenizer model (only required once)

sentences = nltk.sent_tokenize(text)

# Step 7: Paraphrase Sentences
paraphrased_sentences = []

for sentence in sentences:
    paraphrase = paraphrase_model(sentence, max_length=30, num_return_sequences=1)
    paraphrased_sentences.append(paraphrase[0]['generated_text'].strip())

# Step 8: Generate Output
output_file = 'generated.txt'

if os.path.exists(output_file):
    # If the file exists, override it.
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(paraphrased_sentences))
else:
    # If the file doesn't exist, create a new one.
    with open(output_file, 'x', encoding='utf-8') as file:
        file.write('\n'.join(paraphrased_sentences))
