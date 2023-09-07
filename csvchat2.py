# Import necessary libraries
import openai
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

# Set your OpenAI API key
openai.api_key = "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"

# Load CSV data
df = pd.read_csv("C:/Users/SurajKannan/OneDrive - Decision Inc/Desktop/Innovation Project/OpenAI/garminfile/GOTOES_FIT.csv")
st.session_state.df = df

# Define a function to generate responses using the OpenAI language model
def generate_response(user_input):
    # Initialize OpenAI API with your API key and set the temperature parameter
    llm = OpenAI(api_token=openai.api_key, temperature=0.4)
    
    # Create a PandasAI instance for conversational responses
    pandas_ai = PandasAI(llm, conversational=True)
    
    # Generate a response based on user input and the loaded DataFrame
    x = pandas_ai.run(st.session_state.df, prompt=user_input)

    # Get the current Matplotlib figure
    fig = plt.gcf()
    
    # Check if the figure has any axes (plots)
    if fig.get_axes():
        # Display the Matplotlib figure in the Streamlit app
        st.pyplot(fig)
    
    # Return the generated response
    return x

# Set the title of the Streamlit app
st.title("ICE Chatbot Assistant")

# Initialize session state for chat history
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Function to get user input
def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

# Get user input
user_input = get_text()

# Process user input and generate a response
if user_input:
    output = generate_response(user_input)
    
    # Store the user input and generated response in the chat history
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
