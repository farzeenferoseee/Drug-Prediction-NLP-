import streamlit as st
import joblib

#Load pickled files
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("Disease Prediction from Drug Review")
st.write("Enter a drug review (symptoms, experiences, etc.), and we'll predict the disease.")

st.markdown("""
**Currently, prediction is available for the following conditions:**
- Depression  
- Diabetes, Type 2  
- High Blood Pressure
""")

#CSS for custom UI
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .stTextArea textarea {
            background-color: #1a1a1a;
            color: white;
        }
        .stButton button {
            background-color: #800080;
            color: white;
        }
        .stSuccess {
            background-color: #333333;
            color: #adff2f;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App for Deployment
user_input = st.text_area("Please provide your drug review", height=200, placeholder="I've had a sore throat, mild fever...")

if st.button("Predict Disease"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)
        pred_label = label_encoder.inverse_transform(prediction)[0]
        # Display result
        st.success(f"🧬 Predicted Disease: **{pred_label}**")
        
        #Display Probabilities
        st.write("Checking if model supports predict_proba...")
        if hasattr(model, "predict_proba"):
            st.write("Yes, model has predict_proba")
            probs = model.predict_proba(input_vectorized)
            st.write(f"Raw probs: {probs}")
            st.write("Model classes:", model.classes_)
            st.write("Probabilities array:", probs)

            class_labels = label_encoder.inverse_transform(model.classes_)
            st.write("Class labels:", class_labels)

            class_probs = dict(zip(class_labels, probs[0]))
            st.markdown("### 🔍 Prediction Probabilities")
            for label, prob in class_probs.items():
                st.markdown(f"- **{label}**: {prob:.2f}")
        
        
