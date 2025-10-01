from sentence_transformers import SentenceTransformer, util
import streamlit as st
import time

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()


faq = {
    "Qui a remporté le Ballon d’Or cette année": "Le ballon d'or 2025 a été remporté par Ousmane DEMBELE.",
    "Qui a remporté le Ballon d’Or masculin cette année": "Le ballon d'or 2025 a été remporté par Ousmane DEMBELE.",
    "Qui a remporté le Ballon d’Or en 2025": "Le ballon d'or 2025 a été remporté par Ousmane DEMBELE.",
    "Qui a remporté le Ballon d’Or féminin en 2025": "Le ballon d'or 2025 féminin a été remporté par Aitana Bonmatí.",
    "Qui a remporté le Ballon d’Or chez dame cette année": "Le ballon d'or 2025 féminin a été remporté par Aitana Bonmatí.",
    "Où s’est déroulée la cérémonie du Ballon d'or": "La cérémonie du Ballon d'Or 2025 s'est déroulée à Paris, au Théâtre du Châtelet",
    "quelle est la date de la cérémonie du Ballon d'or": "La cérémonie du ballon d'or 2025 a eu lieu le 22 septembre 2025. ",
    "Qui a remporté le Trophée Kopa ": "Le Trophée Kopa 2025 a été remporté par Lamine Yamal Nasraoui Ebana, couramment appelé Lamine Yamal.",
    "bonjour comment tu vas": "Bonjour ! Comment puis-je t’aider ?",
    "salut": "salut ! Comment puis-je t’aider ?",
    "Qui es tu": "Je suis une petite IA capable de repndre à certaines question concernant le Ballon d'Or 2025."
}



synonymes = {
    "bo": "ballon d’or",
    "ballon dor": "ballon d’or",
    "bdor": "ballon d’or"
}

questions = list(faq.keys())
reponses = list(faq.values())

questions_embeddings = model.encode(questions, convert_to_tensor=True)


def type_writer(text, speed=0.05):
    """Affiche du texte caractère par caractère un peu comme CHATGPT"""
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(typed_text)
        time.sleep(speed)

def normaliser_question(question: str) -> str:
    question = question.lower()
    for abbr, full in synonymes.items():
        if abbr in question:
            question = question.replace(abbr, full)
    return question

def detecter_genre(question: str):
    q = question.lower()
    if "homme" in q or "masculin" in q:
        return "homme"
    if "femme" in q or "féminin" in q or "dame" in q:
        return "femme"
    if "date" in q or "jour" in q or "quand" in q:
        return "date"
    return None


def get_response(user_question):
    user_question = normaliser_question(user_question)
    user_question_normalized = user_question.strip().lower()

    genre = detecter_genre(user_question)

    user_question_embedding = model.encode(user_question_normalized, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(user_question_embedding, questions_embeddings)[0]
    best_score_idx = scores.argmax().item()

    # Trier les résultats
    best_idx = scores.argmax().item()
    best_question = questions[best_idx]
    best_answer = faq[best_question]

    # Si la question mentionne "homme" → forcer une réponse masculine
    if genre == "homme":
        for q in questions:
            if "masculin" in q.lower() or "homme" in q.lower():
                return faq[q]

    # Si la question mentionne "femme" → forcer une réponse féminine
    if genre == "femme":
        for q in questions:
            if "féminin" in q.lower() or "dame" in q.lower():
                return faq[q]

    # Si la question mentionne "date"
    if genre == "date":
        for q in questions:
            if "date" in q.lower():
                return faq[q]

    if scores[best_score_idx] >= 0.5:
        return reponses[best_score_idx] # Retourne la réponse correspondante
    else:
        return "Je suis désolé, je n'ai pas la réponse à cette question pour le moment. \nJe suis concu pour repondre aux questions sur le Ballon d'or 2025."


st.set_page_config(page_title="Chatbot Ballon d'Or", layout="centered")

st.title("Chatbot Ballon d'Or 2025")
st.write("Posez vos questions sur le Ballon d'Or 2025 !")

# Historique des messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Afficher l’historique
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Tapez votre question ici...")

if user_input:
    # Ajouter question de l’utilisateur
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Afficher la question de l’utilisateur
    with st.chat_message("user"):
        st.markdown(user_input)

    # Générer la réponse
    answer = get_response(user_input)

    # Ajouter réponse de l’IA
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    

    # Effet écriture progressive uniquement pour la dernière réponse
    with st.chat_message("assistant"):
        type_writer(answer, speed=0.03)