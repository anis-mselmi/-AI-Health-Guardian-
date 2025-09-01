import streamlit as st
from PIL import Image
import torch
import clip
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


with open("models/chat_model.pkl", "rb") as f: 
    chat_model, vectorizer, df = pickle.load(f)

st.title("🩺 AI Health Guardian")


uploaded_file = st.file_uploader("Upload an X-ray/MRI image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    
    text_labels = [
        "COVID-19", "Heart Enlargement", "Normal"
    ]
    text_tokens = clip.tokenize(text_labels).to(device)
    image_input = preprocess(image)
    if not torch.is_tensor(image_input):
        image_input = transforms.ToTensor()(image_input)
    image_input = image_input.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    st.subheader("📸 Image Analysis Results")
    for i, label in enumerate(text_labels):
        st.write(f"{label}: {probs[0][i]*100:.2f}%")
    
    
    fig = plt.figure(figsize=(6,4))
    plt.bar(text_labels, probs[0]*100, color=['skyblue','salmon','lightgreen','orange'])
    plt.ylabel("Probability (%)")
    plt.title("CLIP Image Analysis")  
    st.pyplot(fig)


user_input = st.text_input("📝 Enter your symptoms (comma-separated)")
if user_input:
    X_new = vectorizer.transform([user_input])
    predicted_disease = chat_model.predict(X_new)[0]
    advice = df[df["disease"] == predicted_disease]["advice"].values[0]
    
    st.subheader("💊 Symptoms Chat Results")
    st.write(f"Predicted Disease: **{predicted_disease}**")
    st.write(f"Advice: {advice}")
