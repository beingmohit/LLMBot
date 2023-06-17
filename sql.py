from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

import streamlit as st
from streamlit_chat import message

db = SQLDatabase.from_uri("mysql+pymysql://root:z8lXiPBKotnX9eEEfTM4@localhost:3306/sst")

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

#print(agent_executor.run("How many recoins does user with email bank@changers.com have?"))



st.set_page_config(
    page_title="Demo - SQL Chatbot",
    page_icon=":robot:"
)

st.title('Demo - SQL Chatbot')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","Hello, how can you help me?", key="input")
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