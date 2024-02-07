from itertools import zip_longest

import numpy as np
import pandas as pd
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from sentence_transformers import SentenceTransformer

import streamlit as st

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

st.write("""
# Chat UI
         
This part of the exercise was definitely the most uncertain for me. I had not worked with Langchain before and had only in a very few instances worked with local LLMs. I knew that something like this could be done in a few lines of code due to the abstractions that Streamlit and Langchain provide but I had never seen or done it in practice.
         
I quickly found it to be incredibly easy to get a local LLM up and running with [Ollama](https://ollama.ai/) which only required two lines 1) `ollama serve` 2) `ollama run llama2:7b-chat`.
         
I then found [a repo](https://github.com/AustonianAI/chatbot-starter/tree/main) that exemplified how to use Streamlit, Langchain and OpenAI in conjunction to build a chat interface. This required only minor modifications to work with Ollama.
         
To be able to use the chat interface with images I had to make a few modifications to the code. Additionally, I also wanted to be able for the user to request images and find similar images to the request. I again used the `sentence-transformers` library to embed the captions in the COCO dataset along with a description of their sentiment. I then stored them in a numpy array and is then able to calculate the cosine similarity when the chat detects that the user is requesting to see an image.
""")


st.write("""
## Chat
""")

chat = ChatOllama(model="llama2:7b-chat")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = "You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer. You are an expert regarding image captioning and their sentiment and would like to assist the human in understanding the images based on their caption and sentiment. You have access to the COCO dataset and can find similar images to the user's request. "

with st.sidebar:
    st.text_area("System prompt:", key='system_prompt', height=250)

def detect_image_request(msg):
    msgs = [
    SystemMessage(
        content="""Answer ONLY the parsable token that is specified in the prompt.

        If the user DOES NOT request an image you should respond with a message that says "<NOREQ>". 
        Here are some examples of NOT requesting an image:
        - "I'd like to know what a cat looks like."
        - "Can you describe a bicycle to me?"
        - "I want to know what a mountain looks like."
        - "Where is the Zebra located?"
        - "Can you tell me anything else about New York?"

        
        If the user DOES request to see an image you should answer "<IMGREQ>".
        Here are some examples of image requests:
        - "Can you show me a cat?"
        - "Find me an image of a bicycle."
        - "I'd like to see a picture of a mountain."
        - "I want to see a photo of a dog."
        - "Find a picture of a woman riding a bike."
        - "Hello, can you find me a photo of a man?"
        """
    ), HumanMessage(content=msg)]
    ai_response = chat(msgs)

    if ai_response.content == "<IMGREQ>":
        return True
    else:
        return False

def image_description(msg):
    msgs = [
    SystemMessage(
        content="""Your job is to describe the image in a way that is useful to the user.
        The image will be described to you and you will need to describe it back to the user.

        Always start your response with "Here's a picture of " and then add a description of the image.

        If the user query and image description DOES NOT match you should provide an explanation as to why the user's query might match the image description.
        """
    ),
    HumanMessage(content=msg)]
    ai_response = chat(msgs)

    return ai_response.content


def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content=st.session_state['system_prompt'])]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            if isinstance(ai_msg, dict):
                zipped_messages.append(AIMessage(content=ai_msg["caption"]))
            else:
                zipped_messages.append(
                    AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages


def generate_response():
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    return ai_response.content



# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""


image_descriptions = pd.read_csv("../src/image_descriptions.csv")
vector_db = np.load("../src/image_descriptions.npy")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_image_file_name(image_id):
    return f"../img_data/coco_val2014/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg"

def similar_image(query):
    query_vector = model.encode(query)
    similarities = [cosine_similarity(query_vector, vector) for vector in vector_db]
    top_most_similar = np.argsort(similarities)[-10:]
    most_similar_index = np.random.choice(top_most_similar)

    most_similar = image_descriptions.iloc[most_similar_index]
    image_id = most_similar["image_id"]
    description = most_similar["description"]
    return get_image_file_name(image_id), description



# Create a text input for user
st.text_input('', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    if detect_image_request(user_query):
        print("\tImage request detected for query:", user_query)
        similar_image_path, similar_image_description = similar_image(user_query)
        print("\tSimilar image path:", similar_image_path)
        print("\tSimilar image description:", similar_image_description)
        ai_image_description = image_description(f"User query: {user_query}\n Image description: {similar_image_description}")
        st.session_state.generated.append({"path": similar_image_path, "caption": ai_image_description})
    else:
        # Generate response
        output = generate_response()

        # Append AI response to generated responses
        st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        with st.chat_message("AI"):
            ai_message = st.session_state["generated"][i]
            if isinstance(ai_message, dict):
                st.image(ai_message["path"])
                st.write(ai_message["caption"])
            else:
                st.write(ai_message)
        # Display user message
        with st.chat_message("user"):
            st.write(st.session_state['past'][i])



# Add credit
st.markdown("""
---
Made with 🤖 by [Austin Johnson](https://github.com/AustonianAI)""")