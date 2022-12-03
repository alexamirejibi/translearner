from transformers import pipeline

data = ["go down and to the left. DOWN, UP, LEFT"]

specific_model = pipeline(model="alexamiredjibi/results")
print(specific_model(data))