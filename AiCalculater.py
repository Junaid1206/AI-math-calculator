# import necessary libraries
from ollama import Tool
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.callbacks import StdOutCallbackHandler

# Set up the Streamlit app
st.set_page_config(page_title="Text to Math problem solver", page_icon=":robot:")
st.title("Text to Math problem solver using Gemini :robot:")

#API key setup
gemini_api_key = st.sidebar.text_input(label="Enter your Google Gemini API Key", type="password")

if not gemini_api_key:
    st.info("Please enter your Google Gemini API Key to proceed.")
    st.stop()

# Initialize the LLM with the provided API key
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_api_key)


# Prompt template for converting text to math problems
wiki = WikipediaAPIWrapper()
WikipediaAPITool = Tool(
    name="Wikipedia",
    description="Useful for when you need to look up a topic or person on Wikipedia",
    func=wiki.run
)

#math tool
llm_math_chain = LLMMathChain.from_llm(llm=llm)
calculaor = Tool(
    name = "Calculator",
    func = llm_math_chain.run,
    description = "A tool for answering math problems and calculations"
)

prompt ="""You are my personal agent task solving users math questions

Questions = st.text_input({question})
"""
question = st.text_input("Enter your text here")
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name = "Reasoning",
    func = llm_chain.run,
    description = "A tool for answering logic-based and reasoning questions"
)

assistant_agents = initialize_agent(
    tools=[WikipediaAPITool, calculaor, reasoning_tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    handle_parsing_errors=True,
    verbose=False
)

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{'role':"assistant", 
                                'content':"How can I help you?"}]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
# Accept user input
if st.button("Find it"):
    if question:
        with st.spinner("Finding the answer..."):
            # Display user message in chat message container
            st.chat_message("user").write(question)
            st.session_state.messages.append({'role': "user", 'content': question})

            # Run agent without callback handler
            response = assistant_agents.run(input=question)

            # Save & display assistant's answer
            st.session_state.messages.append({'role': "assistant", 'content': response})
            st.chat_message("assistant").write(response)
    else:
        st.warning("Please enter a question to proceed.")

