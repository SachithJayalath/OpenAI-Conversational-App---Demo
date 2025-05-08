import warnings

# Suppress all deprecation warnings globally
warnings.simplefilter("ignore", DeprecationWarning)

import streamlit as st
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import logging
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)
    
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)

load_dotenv(override=True)

# 4. Setup LLMChain & prompts
thinking_model = ChatOpenAI(temperature=1, model="o4-mini-2025-04-16")
conversational_model = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

template_thinking_model = """
You are a middle AI agent who works in the middle of a powerplant company and their conversatonal AI who is the end face who will report this insights to the user in natural language.
I will share the ground level report of the account balances for the month january of the year 2025 of the powerplant company and the user's message. You will follow ALL of the rules below:

1/ You should check and return all the data in text if it is necessary to answer the user's message.

2/ You should analyse or do any necessary calculations and return them as well but very precisely menetioning what this exact value means.

3/ Do not return guides to do calculations or get certain information as the conversation agent (AI) doesn't have any access to the raw data of reporting. Always do all the necessary calculations and return the results.

4/ If the given user message is completely irrelevant to the given report or the data, then you should only return "IRRELEVANT" and nothing else. Not even a word of saying that it is irrelevant. Just output "IRRELEVANT" nothing else.

5/ Try to give as much as context as possible as the conversational AI agent's response will completely depend on the response you give. When providing numbers or calculation results always provide the Ground Level attributes you got from the report as reference so it would be easier for the conversational AI agent to explain it to the user. But always give the final result or the total amount first and then go into details.

6/ If the user's message is asking for a comparison or analysis of the previous month's values or the previous year's values always use the variance values related to that and the variance percentage as well. The variance is the difference between the current actual value and the previous (year/month) actual value, and the variance percentage is the variance divided by the previous value, multiplied by 100.

7/ If the question is non related to previous year or the previous month only use the actual values (current month) for generating the output no need to get the variance values for the output.  

this is the user's message ; {message}

this the ground level report of the account balance of the company for january 2025 in csv format.
Regarding this report use the following keywords when referring, PM - previous month which is December 2024, PY - previous year which is January 2024
; {gl_report_jan_2025}

You do not have to be emotional or natural language in your response, you should be very precise and technical in your response as you are communicating with another AI.
"""

template_thinking_model_NEW = """
You are a middle AI agent who works in the middle of a powerplant company and their conversatonal AI who is the end face who will report this insights to the user in natural language.
I will share the ground level report of the account balances for the month january of the year 2025 of the powerplant company and the user's message. You will follow ALL of the rules below:

1/ You should check and return all the data in text if it is necessary to answer the user's message.

2/ You should analyse or do any necessary calculations and return them as well but very precisely menetioning what this exact value means.

3/ Do not return guides to do calculations or get certain information as the conversation agent (AI) doesn't have any access to the raw data of reporting. Always do all the necessary calculations and return the results.

4/ If the given user message is completely irrelevant to the given report or the data, then you should only return "IRRELEVANT" and nothing else. Not even a word of saying that it is irrelevant. Just output "IRRELEVANT" nothing else.

5/ Try to give as much as context as possible as the conversational AI agent's response will completely depend on the response you give. When providing numbers or calculation results you can provide the raw data you got from the report as reference so it would be easier for the conversational AI agent to explain it to the user.

this is the user's message ; {message}

this the ground level report of the account balance of the company for january 2025 in csv format and it has been broken down into 5 parts of categories as following.
Regarding this report use the following keywords when referring, PM - previous month which is December 2024, PY - previous year which is January 2024.
The symbol ">" means that the right side category is within the set of the left side.

I.) Total Assets > Current Assets ;

{gl_25jan_current_assets}

II.) Total Assets > Non Current Assets ;

{gl_25jan_non_current_assets}

III.) Total Equity and Liabilities > Equity ;

{gl_25jan_equity}

IV.) Total Equity and Liabilities > Current Liabilities ;

{gl_25jan_current_liabilities}

V.) Total Equity and Liabilities > Non Current Liabilities ;

{gl_25jan_non_current_liabilities}

You do not have to be emotional or natural language in your response, you should be very precise and technical in your response as you are communicating with another AI.
"""

template_conversation_model = """
You are a conversational AI assistant who works with structered data of a powerplant company and report insights in natural language to the users, who will be the management level of the company.
I will share the relevant data that I got filtered from a ground level report of the powerplant company only regarding the month january of 2025 which was generated by a middle thiking AI agent [You should never expose about this thinking model to the user], and you will follow ALL of the rules below:

1/ You should always be conversational and friendly as you are handling the end user who is the management level of the company directly. But do not greet unnecessarily in the start go staright to the point.

2/ You should make a proper structure of the response and make it very easy to read and understand for the user. Make sure to explain if something is complex or technical in a very simple way.

3/ You will be provided the domain knowledge of the powerplant company, so always make sure to blend this domain knwoledge with the data you are provided ONLY if necessary. The user's message can either be regarding the data on the report of the given period or either about the company and domain knowledge itself. If the user's message is completely abou this the domain/company answer this from refering to the brief given.

4/ If you get the response "IRRELEVANT" from the middle AI agent that means the middle agent has decided that the user's message is completely irrelevant to the given report or the data.
In this case explain to the user that the message is irrelevant and you are not able to provide any insights or information regarding this message as you only have access to the financial data of the january 2025 and the compared data of the previous month and the previous year to that period but the user's message also can be on the company or the domain knowledge if so that becomes an exception.
If the user's message is also irrelevant to any of this two and nothing related to the topic, then just simply tell the user that you are unable to help with this and refer him other cloud based, popular AI tools which can help him with this if that is relevant for the user's message.

5/ If you are giving a summary of something, like a total amount of a certain category that user specify, give a total amount or the sum that user has asked first but in this case give some key break down of which sub categories you used WITH NUMBERS related to the each category here secondly. giving numbers in this breakdown is really important if there is any.
If this happenes also try to give the percentage in number next to the actual number as well. But if this breakdown goes very long in context give atleast 5 to 6 points and say etc in natural language.

this is the user's message ; {message}

Below is the relevant report data the program received after going through the ground level report data. This was generated by a middle thinking AI agent [You should never expose about this thinking model to the user]:
{relevant_data}

Here is the brief of the powerplant company and the domain knowledge of the company. Do not give text straight from this just only understand this an explain as an conversational assistant ; {acwa_company_brief}
"""

prompt_th = PromptTemplate(
    input_variables=["message", "gl_report_jan_2025"],
    template=template_thinking_model
)

# comment below and uncomment the above for recovery to the full report reading
# prompt_th = PromptTemplate(
#     input_variables=["message", "gl_25jan_current_assets", "gl_25jan_non_current_assets", "gl_25jan_equity", "gl_25jan_current_liabilities", "gl_25jan_non_current_liabilities"],
#     template=template_thinking_model
# )

prompt_co = PromptTemplate(
    input_variables=["message", "relevant_data", "acwa_company_brief"],
    template=template_conversation_model
)

chain_th = LLMChain(llm=thinking_model, prompt=prompt_th)
chain_co = LLMChain(llm=conversational_model, prompt=prompt_co)

# 5. Retrieval augmented generation
def generate_response_for_thinking(message):
    with open("gl-report-25-january.txt", "r") as file:
        gl_report_jan_2025 = file.read()
    response = chain_th.run(message=message, gl_report_jan_2025=gl_report_jan_2025)
    return response

# comment below and uncomment the above for recovery to the full report reading
# def generate_response_for_thinking(message):
#     with open("gl-report-25-jan-breakdown/totalassets-currentassets.txt", "r") as file:
#         gl_25jan_current_assets = file.read()
#     with open("gl-report-25-jan-breakdown/totalassets-noncurrentassets.txt", "r") as file:
#         gl_25jan_non_current_assets = file.read()
#     with open("gl-report-25-jan-breakdown/totaleuqity.txt", "r") as file:
#         gl_25jan_equity = file.read()
#     with open("gl-report-25-jan-breakdown/totalliabilities-currentliabilities.txt", "r") as file:
#         gl_25jan_current_liabilities = file.read()
#     with open("gl-report-25-jan-breakdown/totalliabilities-noncurrentliabilities.txt", "r") as file:
#         gl_25jan_non_current_liabilities = file.read()
#     response = chain_th.run(message=message, gl_25jan_current_assets=gl_25jan_current_assets, gl_25jan_non_current_assets=gl_25jan_non_current_assets, gl_25jan_equity=gl_25jan_equity, gl_25jan_current_liabilities=gl_25jan_current_liabilities, gl_25jan_non_current_liabilities=gl_25jan_non_current_liabilities)
#     return response

def generate_response_for_convo(message, relevant_data):
    with open("acwa_company_brief.txt", "r") as file:
        acwa_company_brief = file.read()
    response = chain_co.run(message=message, relevant_data=relevant_data, acwa_company_brief=acwa_company_brief)
    return response

def on_submit():
    st.session_state["show_loading"] = True
    st.session_state["show_result"] = False

def return_example(idx):
    examples = {
        1: "What is the total value of my current assets for January 2025?",
        2: "Can you provide a summary of staff expenditures for January 2025?",
        3: "Can you provide an analysis of the 'PIF - Public Investment Fund' based on the available information?",
        4: "Can you provide a breakdown of all the Current Liability categories we have?",
        5: "Give me a short progress overview about my company"
    }
    st.session_state["message"] = examples.get(idx, "")
    st.session_state["show_result"] = False

# 6. Streamlit App
def main():

    if "question_count" not in st.session_state:
        st.session_state["question_count"] = 0

    st.set_page_config(
        page_title="ACWA Conversational Assistant :satellite::milky_way:", page_icon=":milky_way:")

    st.header("ACWA Power :satellite::milky_way:")

    if "message" not in st.session_state:
        st.session_state["message"] = ""

    message = st.text_area("type", key="message", label_visibility="collapsed", height=150)

    # Add empty space before buttons
    st.write("")
    
    # First row of buttons
    col1, col2, col3 = st.columns([0.7, 1, 1])
    with col1:
        st.button("View Current Assets", key="btn1", on_click=lambda: (return_example(1), on_submit()), use_container_width=True)
    with col2:
        st.button("Breakdown of Current Liabilities", key="btn4", on_click=lambda: (return_example(4), on_submit()), use_container_width=True)
    with col3:
        st.button("Analyze PIF Account for 2025 Jan", key="btn3", on_click=lambda: (return_example(3), on_submit()), use_container_width=True)
    

    # Second row of buttons
    col4, col5 = st.columns([1, 1])
    with col4:
        st.button("Staff Spending Summary (Jan 2025)", key="btn2", on_click=lambda: (return_example(2), on_submit()), use_container_width=True)
    with col5:
        st.button("Brief Progress Overview of the Company", key="btn5", on_click=lambda: (return_example(5), on_submit()), use_container_width=True)

    # Enter button row
    col6, = st.columns([0.6])
    with col6:
        st.button("Enter", key="submit", type="primary", on_click=on_submit, use_container_width=True)

    # Add empty space after buttons
    st.write("")

    if st.session_state["message"] and st.session_state["show_loading"]:
        # st.write("Generating best practice snowflake :snowflake: conversion...")
        message = st.session_state["message"]
        if message:
            print()
            print("-"*100)
            st.session_state["question_count"] += 1
            print(f"\nQEUSTION ID : Q{st.session_state['question_count']}")
            print("QUESTION :", message)
            result_th = generate_response_for_thinking(message)
            print("\nRESULT OF THE THINKING MODEL : \n")
            print(result_th)
            result_co = generate_response_for_convo(message, result_th)
            st.session_state["result"] = result_co
            st.session_state["show_result"] = True
            st.session_state["show_loading"] = False

    # Display result if available
    if "show_result" not in st.session_state or "show_loading" not in st.session_state:
        st.session_state["show_loading"] = False
        st.session_state["show_result"] = False

    if st.session_state["show_result"]:
        st.info(st.session_state["result"])

if __name__ == '__main__':
    main()