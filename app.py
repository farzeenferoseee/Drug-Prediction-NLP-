import streamlit as st
import joblib

#Load pickled files
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

#delete later
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
st.write("Label Map:", label_map)

# Streamlit UI
st.title("Disease Prediction from Drug Review")
st.write("Enter a drug review (symptoms, experiences, etc.), and we'll predict the disease.")

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
        prediction = model.predict([input_vectorized])
        pred_label = label_encoder.inverse_transform(prediction)[0]
        # Display result
        st.success(f"🧬 Predicted Disease: **{prediction}**")

        st.markdown(f"""
            ### 📝 Your input:
            `{user_input}`

            ### 🧬 Your result:
            **{pred_label}**
        """)
