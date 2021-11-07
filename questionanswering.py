import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline
import streamlit.components.v1 as components

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
 
 <style>
 .jumbotron{
       background-image: linear-gradient(navy, lightblue, navy);
       }
 .display-6{
           display:flex;
           justify-content: center;
           color:white;
           font-weight: bold;
            background-image: linear-gradient(black, lightblue, navy);
       }
    p{
        display:flex;
        justify-content: center;
        font-weight: bold;
 }
 </style>
<div class="jumbotron">
  <h2 class="display-6">Q&A USE MODEL BERT</h2>
  <p class="lead">Trouvez la réponse dans votre article.</p>
  <hr class="my-2">
  <p>Pour plus d'informations et doutes cliquez sur le bouton ci-dessous.</p>
  <p class="lead">
    <a class="btn btn-primary btn-lg" href="https://huggingface.co/transformers/search.html?q=bert&check_keywords=yes&area=default" target="_blank" role="button">Learn more</a>
  </p>
</div>
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """,
    height=260,
)


@st.cache(allow_output_mutation=True)
# function pour charger ou telecharger le model BERT
def bert_question_answer():
    # ,force_download=True
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    nlp_pipe = pipeline('question-answering', tokenizer=tokenizer, model=model)
    return nlp_pipe


# Appele de la function pipeline
npl_pipe = bert_question_answer()

# integration de zones de text utiliser le
st.header("Veuillez poser des questions en fonction de votre texte")
add_text_sidebar = st.sidebar.title("NATURAL LANGUAGE PROCESSING")
add_text_sidebar = st.sidebar.text("JE SUIS UN ROBOT")
add_text_sidebar = st.sidebar.image('img9.jpg')
add_text_sidebar = st.sidebar.text(""" Salut! Je peux vous aider?
Je suis un robot programmé par l'ingénieur 
Daniel Paulino, je suis là pour répondre à 
vos questions concernant les textes en portugais,
français, darija, anglais, arabe et espagnol""")
add_text_sidebar = st.sidebar.image('img7.jpg')
add_text_sidebar = st.sidebar.text(
    """ Ensemble, nous allons changer le monde.""")

articles = st.text_area("Pease enter your article")
quest = st.text_input("Ask your question based on the article")
button = st.button("Answer")
# spinner pour la recherche de la reponse
with st.spinner("Finding answer..."):
    if button and articles:
        answers = npl_pipe(question=quest, context=articles)
        st.success(answers["answer"])
