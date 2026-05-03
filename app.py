import clip
import torch
from PIL import Image
import os

# Pick device: GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model + preprocessing pipeline
model, preprocess = clip.load("ViT-B/32", device=device)

# Ask user for a text prompt (mood)
mood = input("Enter your mood:")

# Folder containing images
image_folder = "images"

# Store (filename, similarity score) pairs
scores = []

# Iterate through all files in the folder
for file in os.listdir(image_folder):
    path = os.path.join(image_folder, file)

    # Preprocess image → tensor [1,3,224,224] → move to device
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    # Tokenize text → tensor [batch_size,77] → move to device
    text = clip.tokenize([mood]).to(device)

    # Disable gradient tracking (inference mode)
    with torch.no_grad():
        # Encode image and text into embeddings
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Cosine similarity between image and text embeddings
        similarity = torch.cosine_similarity(image_features, text_features)

    # Save filename + similarity score
    scores.append((file, similarity.item()))

# Sort scores descending (best match first)
scores.sort(key=lambda x: x[1], reverse=True)

# Pick best image
best_image = scores[0][0]
best_path = os.path.join(image_folder, best_image)

# Show best match
img = Image.open(best_path)
print("Best Match:", best_image)
img.show()

# Print top 5 matches
print("\nBest Matching Images:\n")
for item in scores[:5]:
    print(item[0], "Score:", item[1])
