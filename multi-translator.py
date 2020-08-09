from transformers import pipeline, MarianTokenizer, MarianMTModel

# Text to translate
sample_text = "I hate processing programming language"

# Using T5 Language models
# Language Translation from EN to DE
translator_en_to_de = pipeline("translation_en_to_de")
print(translator_en_to_de(sample_text))

# Language Translation from EN to FR
translator_en_to_fr = pipeline("translation_en_to_fr")
print(translator_en_to_fr(sample_text))

# Language Translation from EN to RO
translator_en_to_ro = pipeline("translation_en_to_ro")
print(translator_en_to_ro(sample_text))

# Using MarianMT Language models
# Language Translation from EN to SK
en_sk_model_name = f'Helsinki-NLP/opus-mt-en-sk'
en_sk_tok = MarianTokenizer.from_pretrained(en_sk_model_name)
en_sk_model = MarianMTModel.from_pretrained(en_sk_model_name)
en_sk_batch = en_sk_tok.prepare_translation_batch([sample_text])
en_sk_gen = en_sk_model.generate(**en_sk_batch)
print(en_sk_tok.batch_decode(en_sk_gen, skip_special_tokens=True))
