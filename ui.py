import streamlit as st
import clip
import torch
from PIL import Image
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

# Page settings
st.set_page_config(page_title="MoodSnap AI", layout="centered")

# Title
st.title("📸 MoodSnap AI")
st.subheader("Smart Story Photo Selector")
st.caption("Type your mood and let AI pick the perfect story photo.")

# Input box
mood = st.text_input(
    "Enter your mood",
    placeholder="e.g. happy tears, calm sunset, savage mood"
)


# Function to find best matching image
def get_best_image(mood):
    
    image_folder = "images"
    scores = []

    for file in os.listdir(image_folder):
        path = os.path.join(image_folder, file)

        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            text = clip.tokenize([mood]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                similarity = torch.cosine_similarity(
                    image_features, text_features
                )

            scores.append((file, similarity.item()))

        except:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# Button click
if st.button("✨ Suggest Photo", use_container_width=True):
    
    if mood:

        with st.spinner("Finding best photo..."):

            scores = get_best_image(mood)

            if len(scores) > 0:

                best_image = scores[0][0]
                best_path = os.path.join("images", best_image)

                st.success("AI found your best matching vibe.")

                st.markdown("---")
                st.markdown("## 🎯 Best Match")
                
                st.image(
                    best_path,
                    caption=best_image,
                    use_container_width=True
                )

                st.markdown("## 🔥 Top 5 Matches")
                
                cols=st.columns(5)
                
                for i, item in enumerate(scores[0:5]):
                    img_path=os.path.join("images",item[0])
                    
                    with cols[i]:
                        st.image(img_path,caption=f"{i+1}", use_container_width=True)

            else:
                st.error("No images found in images folder.")

    else:
        st.warning("Please enter your mood.")