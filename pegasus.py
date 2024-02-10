from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


def sumarizador(texto_input):
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer(texto_input, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=1024, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


class PegasusScorer:
    def __init__(self, model_name="google/pegasus-xsum"):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

    def sentence_score(self, texto):
        sentencas = texto.split('.')
        sentence_scores = []

        for sentenca in sentencas:
            input_ids = self.tokenizer.encode(sentenca, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, decoder_input_ids=input_ids)
            logits = outputs.logits
            score = torch.mean(logits).item()
            sentence_scores.append(score)

        # Normalizar os scores entre -1 e 1
        max_score = max(sentence_scores)
        min_score = min(sentence_scores)
        normalized_scores = [(2 * (score - min_score) / (max_score - min_score)) - 1 for score in sentence_scores]
        for idx, (sentenca, score) in enumerate(zip(sentencas, normalized_scores)):
            print(f"{idx + 1}: {sentenca.strip()} - Score : {score}")
        return normalized_scores