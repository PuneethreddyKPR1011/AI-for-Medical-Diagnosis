import streamlit as st
import pandas as pd
import joblib
import os
from groq import Groq
from setfit import SetFitModel
# Assuming 'datasets' might not be directly needed for prediction/chat,
# but keeping it just in case SetFitModel relies on it internally or for type hints.
# from datasets import Dataset # Uncomment if needed by SetFitModel internally
import torch # Explicitly import torch if needed

# --- Configuration ---
SETFIT_MODEL_PATH = #Path to your setfit_disease_model Folder
LABEL_MAPPING_PATH = #Path to your LABEL_MAPPINGdisease_label_mapping.pkl file
GROQ_MODEL_NAME = "llama3-8b-8192" # Recommended basic model, or use your original choice 

# --- Helper Functions & Model Loading ---

# Cache the SetFit model and label mapping to load only once
@st.cache_resource
def load_setfit_model_and_mapping():
    """Loads the SetFit model and label mapping."""
    # Check if paths exist before attempting to load
    if not os.path.exists(SETFIT_MODEL_PATH):
         st.error(f"Error: SetFit model directory not found at '{SETFIT_MODEL_PATH}'. Please check the path.")
         return None, None
    if not os.path.exists(LABEL_MAPPING_PATH):
        st.error(f"Error: Label mapping file not found at '{LABEL_MAPPING_PATH}'. Please check the path.")
        return None, None

    try:
        # Ensure the model is loaded from the correct directory structure expected by SetFitModel
        # If the model files (.bin, config.json, etc.) are directly in SETFIT_MODEL_PATH, this is correct.
        model = SetFitModel.from_pretrained(SETFIT_MODEL_PATH, local_files_only=True)
        label_mapping = joblib.load(LABEL_MAPPING_PATH)
        # Add a check to ensure label_mapping is a dictionary as expected
        if not isinstance(label_mapping, dict):
             st.error(f"Error: Loaded label mapping is not a dictionary. Check the content of '{LABEL_MAPPING_PATH}'.")
             return model, None # Return model if loaded, but indicate mapping issue
        return model, label_mapping
    except Exception as e:
        # Provide more specific feedback if possible
        st.error(f"Error loading SetFit model or mapping: {e}")
        st.error(f"Please ensure the SetFit model files are correctly placed in '{SETFIT_MODEL_PATH}' and the label mapping pickle file is valid at '{LABEL_MAPPING_PATH}'.")
        return None, None

def predict_disease(symptom_description, model, label_mapping):
    """Predicts disease based on symptom description using the loaded model."""
    if not model:
        st.error("SetFit Model is not loaded. Cannot predict.")
        return "Model not loaded."
    if not label_mapping:
        st.error("Label Mapping is not loaded. Cannot interpret prediction.")
        return "Label mapping not loaded."

    try:
        # Ensure input is a list for SetFit model prediction
        if not isinstance(symptom_description, list):
            symptom_list = [str(symptom_description)] # Convert to string just in case
        else:
            symptom_list = symptom_description

        preds = model.predict(symptom_list)

        # Assuming preds returns a list or tensor of predicted indices
        if isinstance(preds, torch.Tensor):
             pred_label_int = int(preds[0].item()) # Use .item() to get Python int from Tensor
        elif isinstance(preds, list):
             pred_label_int = int(preds[0]) # Assume list contains integer indices
        else:
             # Handle unexpected prediction format
             st.error(f"Unexpected prediction format: {type(preds)}. Expected list or tensor.")
             return "Prediction interpretation failed."

        predicted_disease = label_mapping.get(pred_label_int, f"Unknown label index: {pred_label_int}")
        return predicted_disease
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Add more context if possible
        st.error(f"Input to model was: {symptom_list}")
        return "Prediction failed."

# Cache the Groq client
@st.cache_resource
def get_groq_client():
    """Initializes and returns the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Try to get from Streamlit secrets if available
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except (AttributeError, KeyError):
             st.error("GROQ_API_KEY not found. Please set it as an environment variable or in Streamlit secrets ([groq] GROQ_API_KEY = 'YOUR_API_KEY').")
             st.info("You can get a Groq API key from: https://console.groq.com/keys")
             st.stop() # Stop execution if API key is missing

    try:
        client = Groq(api_key=api_key)
        # Optional: Add a test ping or simple API call here to verify the key works
        # client.models.list() # Example check - might incur cost/quota usage
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.info("Please ensure your Groq API key is correct and active.")
        st.stop()

# --- Streamlit App Layout ---

st.set_page_config(page_title="MedBot Assistant", layout="wide", initial_sidebar_state="collapsed") # Collapse sidebar initially if not used
st.title("🩺 MedBot: AI Medical Assistant")
st.markdown(
    """
    Welcome to MedBot! This tool offers two features:
    1.  **Disease Prediction:** Enter your symptoms, and the AI will suggest a possible condition based on its training data.
    2.  **Chat with MedBot:** Ask health-related questions or discuss the prediction results with our AI assistant.

    **Important:** MedBot provides information and AI insights but is **not a substitute for professional medical advice**. Always consult a qualified healthcare provider for diagnosis and treatment.
    """
)
st.divider()

# --- Load Models and Client ---
# Use placeholders while loading
model_status = st.empty()
client_status = st.empty()

with model_status.status("Loading prediction model...", expanded=False):
    setfit_model, label_mapping = load_setfit_model_and_mapping()
if setfit_model and label_mapping:
    model_status.success("Prediction model loaded successfully.")
else:
    model_status.error("Prediction model failed to load. Check error messages above.")

with client_status.status("Initializing chatbot...", expanded=False):
    groq_client = get_groq_client()
if groq_client:
     client_status.success("Chatbot initialized successfully.")
else:
     # Error message handled within get_groq_client, no need to repeat unless adding more info
     client_status.error("Chatbot failed to initialize. Check error messages above.")


# Only show the prediction section if the model loaded successfully
if setfit_model and label_mapping:
    st.subheader("1. Disease Prediction Tool")
    st.markdown("Enter symptoms or a description to get a possible disease prediction.")

    symptom_input = st.text_area("Enter symptoms here:", height=100, key="symptom_input", placeholder="e.g., high fever, persistent cough, headache")

    if st.button("Predict Disease", key="predict_button", type="primary"):
        if symptom_input.strip(): # Check if input is not just whitespace
            with st.spinner("Analyzing symptoms and predicting..."):
                prediction = predict_disease(symptom_input, setfit_model, label_mapping)
                st.success(f"**Possible Condition:** {prediction}")
                st.caption("_Disclaimer: This prediction is based on an AI model and patterns in data. It is **not** a medical diagnosis. Please consult a doctor._")
        else:
            st.warning("Please enter symptoms or a description.")
else:
    st.subheader("1. Disease Prediction Tool")
    st.warning("The disease prediction model could not be loaded. This feature is currently unavailable.")

st.divider() # Visual separator

# --- Chatbot Section ---
st.subheader("2. Chat with MedBot")

# Only enable chat if the Groq client loaded successfully
if groq_client:
    st.markdown("Ask MedBot about symptoms, general health concerns, or discuss the prediction results if you used the tool above.")

    # Initialize chat history in session state
    system_message = {
        "role": "system",
        "content": (
            "You are MedBot, an intelligent, caring AI medical assistant integrated into a Streamlit application. "
            "This application includes a separate tool that attempts to predict a possible disease based on user-input symptoms using a SetFit model. "
            "Your role is to be conversational, empathetic, and informative regarding general health topics, symptoms, and potential conditions. "
            "You can help users understand the *possible* implications of symptoms or the prediction they received from the *other* tool, but you MUST emphasize that neither you nor the prediction tool provide a diagnosis. "
            "Explain medical concepts clearly. "
            "Crucially, ALWAYS remind users that you are an AI assistant, not a healthcare professional, and that they MUST consult with a qualified doctor or healthcare provider for any real medical concerns, diagnosis, or treatment. Do not give medical advice or instructions. Focus on providing information and encouraging consultation with professionals."
            "Keep your responses concise and easy to understand."
        )
    }

    if "messages" not in st.session_state:
        st.session_state.messages = [system_message]

    # Display chat messages from history
    for message in st.session_state.messages:
        # Don't display the system message in the chat UI
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input using chat_input
    if prompt := st.chat_input("Ask MedBot... (e.g., 'What are common cold symptoms?', 'Tell me about hypertension')"):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Prepare messages for the API - exclude the system prompt if the API handles it implicitly or based on its documentation
                # For Groq/OpenAI API style, include the system prompt.
                api_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                # Add a thinking indicator
                with message_placeholder.status("MedBot is thinking...", expanded=False):
                    response = groq_client.chat.completions.create(
                        model=GROQ_MODEL_NAME,
                        messages=api_messages,
                        # stream=True # Use streaming for better UX
                        stream=False # Keep False for simplicity unless you implement chunk handling
                    )

                # Handle non-streaming response
                if not isinstance(response, str): # Check if it's the expected object
                    full_response = response.choices[0].message.content
                    message_placeholder.markdown(full_response)
                else:
                    # Handle unexpected string response if API behaves differently
                     message_placeholder.markdown("Received an unexpected response format from the chat service.")


                # --- Streaming Response Handling (Optional) ---
                # If stream=True:
                # response_stream = groq_client.chat.completions.create(
                #     model=GROQ_MODEL_NAME,
                #     messages=api_messages,
                #     stream=True
                # )
                # for chunk in response_stream:
                #     if chunk.choices[0].delta.content is not None:
                #         full_response += chunk.choices[0].delta.content
                #         message_placeholder.markdown(full_response + "▌") # Add cursor effect
                # message_placeholder.markdown(full_response) # Final response without cursor
                # -------------------------------------------

            except Exception as e:
                st.error(f"An error occurred with the chatbot: {e}")
                full_response = "Sorry, I encountered an error while trying to respond. Please ensure your Groq API key is valid and has quota available. You might try again later."
                message_placeholder.markdown(full_response)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add a button to clear chat history
    if len(st.session_state.messages) > 1: # Show clear button only if there's history beyond system prompt
         if st.button("Clear Chat History", key="clear_chat"):
             st.session_state.messages = [system_message] # Reset history
             st.success("Chat history cleared.")
             st.rerun() # Rerun to update the display immediately

else:
    st.subheader("2. Chat with MedBot")
    st.warning("The chatbot could not be initialized. This feature is currently unavailable. Please check the API key configuration.")


# Add a persistent footer disclaimer
st.divider()
st.caption(
    """
    ---
    **🩺 MedBot Assistant Disclaimer:** This tool provides AI-generated information and predictions for informational purposes only.
    It is **not** a substitute for professional medical advice, diagnosis, or treatment.
    **Always seek the advice of your physician or other qualified health provider** with any questions you may have regarding a medical condition.
    Never disregard professional medical advice or delay in seeking it because of something you have read or seen in this application.
    """
)
