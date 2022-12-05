from transformers import pipeline

data = ["go down and to the left. LEFT, LEFT, DOWN, DOWN"]

specific_model = pipeline(model="alexamiredjibi/traj-classifier-recency")

print(specific_model(data))