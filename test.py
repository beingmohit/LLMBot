import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import streamlit as st
from streamlit_chat import message

documents = SimpleDirectoryReader('./data/paul_graham_essay').load_data()

index = VectorStoreIndex.from_documents(documents=documents)

tools = [
    Tool(
        name = "LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
        return_direct=True
    ),
]

# set Logging to DEBUG for more detailed outputs
memory = ConversationBufferMemory(memory_key="chat_history")

llm = ChatOpenAI(temperature=0)

agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)

# print(agent_executor.run(input="hi, i am bob"))

# print(agent_executor.run(input="What did the author do growing up?"))

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

st.title('Quickstart App')
st.header("Streamlit Chat - Demo")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 

user_input = get_text()

if user_input:
    output = agent_executor.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# def generate_response(input_text):
#   output = agent_executor.run(input=input_text)
#   st.info(output)

# with st.form('my_form'):
#   text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#   submitted = st.form_submit_button('Submit')
  
#   if submitted:
#     generate_response(text)