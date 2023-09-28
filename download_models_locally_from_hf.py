from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


MODELS = [
        "roberta-base",
        "microsoft/deberta-base",
              ]


MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(f"{MODEL_NAME}")

model.save_pretrained(f"{MODEL_NAME}")

