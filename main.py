"""Application Streamlit de prédiction du risque de crédit.

Lancer après l'entraînement :
    streamlit run main.py
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "credit_model.pkl"
st.set_page_config(page_title="Risque de crédit", page_icon="🏦", layout="centered")


@st.cache_resource
def load_bundle():
    """Charge le modèle sauvegardé une seule fois par session Streamlit."""
    return joblib.load(MODEL_PATH)


st.title("🏦 Prédiction du risque de crédit")
st.caption("Estimation du risque de refus à partir des caractéristiques du client.")

if not MODEL_PATH.exists():
    st.error("Modèle introuvable. Lancez d'abord : `python train_model.py`")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]

with st.sidebar:
    st.header("À propos")
    st.write(
        "La prédiction utilise une Random Forest optimisée. "
        "La classe 1 correspond à un prêt refusé (proxy de risque)."
    )
    with open(MODEL_PATH, "rb") as model_file:
        st.download_button(
            "Télécharger le modèle",
            data=model_file.read(),
            file_name="credit_model.pkl",
            mime="application/octet-stream",
        )

with st.form("credit_form"):
    st.subheader("Informations du client")
    dependents = st.number_input("Nombre de personnes à charge", 0, 20, 0)
    education = st.selectbox("Niveau d'éducation", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Travailleur indépendant ?", ["No", "Yes"])

    st.subheader("Situation financière")
    income = st.number_input("Revenu annuel", min_value=0.0, value=5_000_000.0, step=100_000.0)
    loan_amount = st.number_input("Montant demandé", min_value=0.0, value=15_000_000.0, step=100_000.0)
    loan_term = st.number_input("Durée du prêt", min_value=1, max_value=50, value=12)
    cibil_score = st.number_input("Score CIBIL", min_value=300, max_value=900, value=650)

    st.subheader("Actifs")
    residential = st.number_input("Valeur des actifs résidentiels", min_value=0.0, value=5_000_000.0, step=100_000.0)
    commercial = st.number_input("Valeur des actifs commerciaux", min_value=0.0, value=3_000_000.0, step=100_000.0)
    luxury = st.number_input("Valeur des actifs de luxe", min_value=0.0, value=5_000_000.0, step=100_000.0)
    bank = st.number_input("Valeur des actifs bancaires", min_value=0.0, value=3_000_000.0, step=100_000.0)
    submitted = st.form_submit_button("Prédire le risque")

if submitted:
    total_assets = residential + commercial + luxury + bank
    client = pd.DataFrame([{
        "no_of_dependents": dependents,
        "education_encoded": int(education == "Graduate"),
        "self_employed_encoded": int(self_employed == "Yes"),
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank,
        "loan_to_income_ratio": loan_amount / income if income else 0,
        "total_assets_value": total_assets,
        "loan_to_assets_ratio": loan_amount / total_assets if total_assets else 0,
    }])[features]

    risk_probability = float(model.predict_proba(client)[0, 1])
    prediction = int(model.predict(client)[0])

    st.divider()
    if prediction == 1:
        st.error(f"Risque de refus élevé : {risk_probability:.1%}")
    else:
        st.success(f"Risque de refus faible : {risk_probability:.1%}")

    st.progress(int(risk_probability * 100))
    st.caption("Résultat indicatif : une décision réelle doit être validée par les règles et analystes de la banque.")

st.divider()
st.subheader("Valeur métier")
st.markdown(
    "- Décision plus rapide et plus cohérente pour les dossiers standards.\n"
    "- Score CIBIL, durée, revenu, montant demandé et actifs contribuent à l'évaluation.\n"
    "- L'outil aide la décision ; il ne remplace pas le contrôle humain ni les exigences réglementaires."
)
