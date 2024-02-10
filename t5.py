from transformers import T5ForConditionalGeneration, T5Tokenizer


def sumarizador(texto_input):
    model_name = "t5-large"  # Você pode escolher diferentes versões do T5, como 't5-base' ou 't5-large'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # O T5 requer que o texto de entrada seja formatado com um prefixo específico, como "summarize: "
    input_ids = tokenizer("summarize: " + texto_input, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Gerando a sumarização
    summary_ids = model.generate(
        input_ids,
        max_length=1500,  # Aumente para um valor maior para sumarizações mais longas
        min_length=75,  # Ajuste conforme necessário para garantir um comprimento mínimo
        length_penalty=2.0,
        num_beams=4,
        early_stopping=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary