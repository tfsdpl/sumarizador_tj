from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

def traduzir_pt_en(texto):
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer.encode(texto, return_tensors="pt", max_length=512, truncation=True)
    translated = model.generate(inputs, max_length=512)
    texto_traduzido = tokenizer.decode(translated[0], skip_special_tokens=True)

    return texto_traduzido



def traduzir_en_pt(texto):
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer.encode(texto, return_tensors="pt", max_length=512, truncation=True)
    translated = model.generate(inputs, max_length=512)
    texto_traduzido = tokenizer.decode(translated[0], skip_special_tokens=True)

    return texto_traduzido


def sumarizador(text):
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=9, max_length=1800, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def sentence_score(texto):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    sentencas = texto.split('.')
    sentence_scores = []
    for sentenca in sentencas:
        inputs = tokenizer(sentenca, return_tensors="pt", max_length=1024, truncation=True, padding=True)
        outputs = model(**inputs)
        score = outputs.logits[0].detach().cpu().numpy().mean()  # Calcula a média dos scores
        sentence_scores.append(score)
    max_score = max(sentence_scores)
    min_score = min(sentence_scores)
    normalized_scores = [(2 * (score - min_score) / (max_score - min_score)) - 1 for score in sentence_scores]

    for idx, (sentenca, score) in enumerate(zip(sentencas, normalized_scores)):
        print(f"{idx + 1}: {sentenca.strip()} - Score : {score}")


