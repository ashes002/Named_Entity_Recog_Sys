# Named_Entity_Recog_Sys

This repository implements a Named Entity Recognition system for the Turkish Language. The code will probably work for the other languages as well if the input data format criterion matches.

The model's architecture is Bi-LSTM + CRF. FastText embeddings are used as a pre-trained word embeddings. The code also supports the variable size input.

# Examples
```
# For the complete example, refer test.py
sentence = (
    "Aziz Nesin 'in yazmış olduğu Nesin Yayınevi tarafından basılan ' Bir Sürgünün Anıları ' "
    "isimli kitap Nesin 'in sürgün yıllarındaki Bursa anılarını anlatıyor ."
)
# We assume that the sentence is already tokenized.
sentence_tokens = sentence.split()
score, tags = predict_sentence(model, sentence_tokens, w2i, idx2tag)

for token, tag in zip(sentence_tokens, tags):
    print("{:<15}{:<5}".format(token, tag))

# Output:
# Aziz           B-PER
# Nesin          I-PER
# 'in            O
# yazmış         O
# olduğu         O
# Nesin          B-ORG
# Yayınevi       I-ORG
# tarafından     O
# basılan        O
# '              O
# Bir            O
# Sürgünün       O
# Anıları        O
# '              O
# isimli         O
# kitap          O
# Nesin          B-PER
# 'in            O
# sürgün         O
# yıllarındaki   O
# Bursa          B-LOC
# anılarını      O
# anlatıyor      O
# .              O
```