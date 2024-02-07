import os

import streamlit as st

st.set_page_config(layout="wide")


st.write("# Laerdal NLP Case")

st.write("""## Goal breakdown

All models available can be used to solve the task. These could be found on HuggingFace, Spacy, LangChain or other ML Libraries.

1. Sentiment: Show in practice how you would apply a machine learning model on the captions to extract the sentiment (negative or positive) of them.
         
2. Topic: Show in practice how you would apply a ML model on the captions to extract the topic(s) that they belong to.
         
3. Repeat step 1+2 with a ML model extracting information from the images valuable for determining the topic and sentiment (a pretrained Vision Language Model, VLM, could be beneficial)
         
4. Compare the results from having captions only and then adding the image understanding.
         
5. With an LLM in Langchain and Streamlit as the UI, write a Chat User Interface, where the user can have a dialog about their images, where the sentiment and topic is helping them and the chatbot to understand the images.
""")


st.write("""## Self-proposed case limitations
         
I thought that the following limitations would suit the case well and make it realistic in terms of time and resources:

1. Use only my MacBook Pro (M2, 16GB RAM) as the computational resource.
    
    1a. I will allow myself to use a GPU cluster at DTU for the image processing part of the case.
    
2. In the spirit of keeping things simple I will attempt to limit model training as much as possible.
         
    2a. Will allow for fine-tuning of models if it is feasible to do on my laptop.

""")