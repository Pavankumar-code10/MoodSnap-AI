import clip
import torch
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

mood = input("Enter your mood: ")

image_folder = "images"

scores = []

for file in os.listdir(image_folder):
    path = os.path.join(image_folder, file)

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = clip.tokenize([mood]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        similarity = torch.cosine_similarity(image_features, text_features)

    scores.append((file, similarity.item()))

scores.sort(key=lambda x: x[1], reverse=True)

best_image = scores[0][0]


best_path = os.path.join(image_folder, best_image)

img = Image.open(best_path)

print("Best Match:", best_image)

img.show()

print("\nBest Matching Images:\n")

for item in scores[:5]:
    print(item[0], "Score:", item[1])
