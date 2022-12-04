from transformers import pipeline

data = ["go down and to the left. DOWN, DOWN, LEFT, LEFT, DOWN, RIGHT, RIGHT"]

specific_model = pipeline(model="alexamiredjibi/trajectory-classifier2")

logits, labels = specific_model(data)
print(logits)
print(labels)