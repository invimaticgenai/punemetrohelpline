from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

openai.api_key = st.secrets["API_KEY"]
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='6348c0b6-eb7a-4b83-8ff9-2c2c6ab11887', environment='gcp-starter')
index = pinecone.Index('pine-chatbot')


def find_match(input1):
    input_em = model.encode(input1).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']


def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most "
               f"relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION "
               f"LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string