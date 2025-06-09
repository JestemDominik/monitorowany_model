import os
import json
import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from dotenv import load_dotenv
from langfuse import Langfuse, observe
from langfuse.openai import openai  # <--- Używamy openai z langfuse

# Wczytanie zmiennych środowiskowych
load_dotenv()

# Inicjalizacja Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Sprawdzenie klucza OpenAI
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    else:
        st.info("Podaj swój klucz OpenAI:")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

# Ustawienie klucza OpenAI
openai.api_key = st.session_state["openai_api_key"]


# Dekorowana funkcja ekstrakcji danych z wiadomości
@observe()
def get_data_from_message(message: str, model="gpt-4o"):
    prompt = """
    You are an assistant for extracting information from text written in Polish.
    You will receive the content and extract the following information:

    <Płeć> - Sex of the person: 'M' for male, 'K' for female.
    <Wiek> - Age of the person.
    <5 km Tempo> - The average tempo (minutes per kilometer) with which the person is running 5 km.

    Return the following dictionary:
    {
        "Płeć": "...",
        "Wiek": ...,
        "5 km Tempo": ...,
    }

    All fields must be present (use null if unknown).
    Return only a valid JSON dictionary — nothing else.

    Example email:
    hej, nazywam się Marek Marucha, jestem mężczyzną, mam 37 lat i biegnę 5km ze średnim tempem 5.23 minut na kilometr

    Expected output:
    {
        "Płeć": "M",
        "Wiek": 37,
        "5 km Tempo": 5.23
    }
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message}
    ]

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        raw_response = response.choices[0].message.content
        return json.loads(raw_response)
    except Exception as e:
        return {"error": str(e)}


# Model PyCaret
@st.cache_resource
def load_marathon_model():
    return load_model("Pipeline_maratończyka")

loaded_model = load_marathon_model()


# UI Streamlit
st.title(':man-running: Sprawdź w jakim czasie byś przebiegł półmaraton:')
user_info = st.text_area('Przedstaw się, powiedz ile masz lat, podaj swoją płeć i średnie tempo z jakim biegniesz 5km:')

if st.button('Sprawdź :stopwatch:'):
    extracted = get_data_from_message(user_info)

    if extracted.get("Płeć") and extracted.get("Wiek") and extracted.get("5 km Tempo"):
        new_data = pd.DataFrame([{
            "Płeć": extracted["Płeć"],
            "Wiek": extracted["Wiek"],
            "5 km Tempo": extracted["5 km Tempo"]
        }])

        prediction = predict_model(loaded_model, data=new_data)
        wynik = prediction['prediction_label'].values[0]
        st.success(f"🏁 Przewidywany czas półmaratonu: {wynik:.2f} sekund")
    else:
        st.error(f"❌ Nie udało się odczytać wszystkich danych. Szczegóły: {extracted.get('error', 'Brak wymaganych danych')}")