import os
from apikey import apikey

import streamlit as st
# from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain   #onnly gives the final output-> SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

from langchain_community.llms import OpenAI
os.environ['OPENAI_API_KEY'] = apikey


st.title("Youtube Script Generator🎈")
prompt = st.text_input("What is your prompt?")
print(prompt)
print(type(prompt))

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template='write me a youtube script based on this title : {title} while leveraging this wikipedia reasearch:{wikipedia_research}'
)


#memory
title_memory = ConversationBufferMemory(input_key='topic',memorykey='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memorykey='chat_history')

llm = OpenAI(temperature=0.5)
title_chain=LLMChain(llm=llm, prompt=title_template,verbose=True,output_key='title',memory=title_memory)
script_chain=LLMChain(llm=llm, prompt=script_template,verbose=True,output_key='script',memory=script_memory)

# sequential_chain=SequentialChain(chains=[title_chain, script_chain],input_variables=['topic'],output_variables=['title','script'], verbose=True)
wiki=WikipediaAPIWrapper()
if prompt:
    title=title_chain.run(prompt)
    wiki_research=wiki.run(prompt)
    script = script_chain.run(title=title,wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('titles history'):
        st.info(title_memory.buffer)
    with st.expander('Script history'):
        st.info(script_memory.buffer)
    with st.expander('wikipedia research'):
        st.info(wiki_research)

