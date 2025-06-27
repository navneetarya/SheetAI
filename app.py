# ==============================================================================
# app.py - Conversational AI Google Sheets Analyst (DEFINITIVE OVERHAUL)
# ==============================================================================

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
from contextlib import redirect_stdout
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import logging
import sys

# --- Import Google/AI Libraries ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from groq import Groq

# --- Setup Detailed Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(page_title="Conversational Sheets Analyst", page_icon="🤖", layout="centered")

# --- Load Environment Variables ---
load_dotenv()


# --- Data Loading and Preparation ---
@st.cache_data(ttl=600)
def get_data_from_sheet(spreadsheet_id, range_name):
    try:
        creds = None
        if os.path.exists('token.json'): creds = Credentials.from_authorized_user_file('token.json',
                                                                                       ['https://www.googleapis.com/auth/spreadsheets.readonly'])
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json',
                                                                 ['https://www.googleapis.com/auth/spreadsheets.readonly'])
                creds = flow.run_local_server(port=0, open_browser=False)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        service = build('sheets', 'v4', credentials=creds)
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])
        if not values: return None

        df = pd.DataFrame(values[1:], columns=values[0])
        logger.info("Performing smart data type conversion.")
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='raise', format='mixed')
            except (ValueError, TypeError):
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    pass
        if 'Units Sold' in df.columns and 'Unit Price' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Units Sold']) and pd.api.types.is_numeric_dtype(df['Unit Price']):
                df['Revenue'] = df['Units Sold'] * df['Unit Price']
                logger.info("Proactively created 'Revenue' column.")
        return df
    except Exception as e:
        logger.error(f"Data loading/authentication failed: {e}", exc_info=True)
        return None


# --- AI Communication and Logic ---
def get_ai_response(system_prompt, user_prompt, model="llama3-8b-8192"):
    try:
        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error communicating with AI: {e}", exc_info=True)
        return None


# --- FIX 1: A TRULY ROBUST, KEYWORD-BASED CLASSIFIER ---
def classify_intent_by_rules(query):
    query_lower = query.lower()
    # Using dictionaries for clarity and easier expansion
    intent_keywords = {
        "greeting": ["hello", "hi", "hey", "how are you"],
        "help_request": ["help", "what can you do", "options", "what charts", "what can i create"],
        "query_suggestion": ["sample", "example", "suggestion", "what should i ask", "give me questions"]
    }
    for intent, keywords in intent_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            logger.info(f"Rule-based intent classified as: {intent}")
            return intent
    logger.info("No rule matched. Defaulting to data_query.")
    return "data_query"


# --- FIX 2: ADVANCED "ONE-SHOT" PROMPTING ---
def generate_python_code(query, df_schema):
    system_prompt = """You are a world-class Python data analyst bot. Your sole purpose is to write clean, executable Python code to answer a user's question about a pandas DataFrame named `df`.
You will be penalized if you respond with anything other than pure Python code.
You must follow these rules:
1.  Your response MUST be ONLY Python code. No text, explanations, or markdown.
2.  You are ONLY allowed to use `pandas` and `matplotlib.pyplot`. Do NOT import any other libraries.
3.  For plots, you MUST use `fig, ax = plt.subplots()` and plot on the `ax` object. The figure variable must be named `fig`.
4.  You MUST NOT include `plt.show()` in your code.
5.  If the user asks about revenue, assume a 'Revenue' column already exists."""

    user_prompt = f"""
Here is a perfect example of how to respond.
---
EXAMPLE
User's question: "Create a bar chart of the total revenue per category"
The DataFrame `df` has this schema:
Category: object
Revenue: float64

Your response (MUST be only this code):
fig, ax = plt.subplots()
revenue_by_category = df.groupby('Category')['Revenue'].sum()
revenue_by_category.plot(kind='bar', ax=ax)
ax.set_title('Total Revenue by Category')
ax.set_ylabel('Total Revenue')
---

Now, perform this task:
User's question: '{query}'
The DataFrame `df` has this schema:
{df_schema}

Your response:
"""
    code = get_ai_response(system_prompt, user_prompt, model="llama3-70b-8192")
    if code:
        code = code.strip()
        if code.startswith("```python"): code = code[9:]
        if code.endswith("```"): code = code.strip()[:-3]
        return code.strip()
    return ""


def generate_query_suggestions(df_schema, context_query):
    system_prompt = "You are a helpful assistant who suggests data analysis questions based on a DataFrame schema. You must only respond with a Python list of strings."

    # Add context if user asked about charts
    if 'chart' in context_query.lower():
        user_prompt = f"The user is interested in charts. Based on this DataFrame schema, create 5 insightful chart-related questions:\n{df_schema}"
    else:
        user_prompt = f"Based on this DataFrame schema, create 5 diverse questions (including calculations and charts):\n{df_schema}"

    response_str = get_ai_response(system_prompt, user_prompt)
    try:
        suggestions = eval(response_str)
        if isinstance(suggestions, list): return suggestions
    except:
        pass
    return ["What is the total 'Revenue'?", "Plot a bar chart of 'Revenue' per 'Region'."]


def is_valid_code(code_string):
    if not code_string: return False
    # A simple check for common pandas/matplotlib patterns. More robust than just keywords.
    return 'df' in code_string or 'plt' in code_string


def execute_code(code, df):
    local_scope = {'df': df, 'pd': pd, 'plt': plt}
    string_io = io.StringIO()
    try:
        with redirect_stdout(string_io):
            exec(code, local_scope)
        printed_output = string_io.getvalue()
        fig = local_scope.get('fig', None)
        if fig:
            if hasattr(fig, 'figure'): fig = fig.figure
            return {"type": "plot", "content": fig}
        elif printed_output:
            return {"type": "text", "content": f"```\n{printed_output}\n```"}
        else:
            # --- FIX 3: HONEST AND CLEAR ERROR FOR NO OUTPUT ---
            return {"type": "error",
                    "content": "I ran the analysis, but it didn't produce a chart or a text answer. Please try rephrasing your question to be more specific."}
    except Exception as e:
        logger.error(f"Code execution failed: {e}\nCode:\n{code}", exc_info=True)
        return {"type": "error", "content": f"An error occurred: {e}\n\n**Generated Code:**\n```python\n{code}\n```"}


# ==============================================================================
# --- STREAMLIT UI ---
# ==============================================================================
st.title("🤖 Conversational Sheets Analyst")

with st.sidebar:
    # Sidebar remains unchanged
    st.header("🔗 Connection Details")
    sheet_id = st.text_input("Google Sheet ID")
    sheet_name = st.text_input("Sheet Name", "Sheet1")
    if st.button("Connect to Sheet", type="primary"):
        with st.spinner("Connecting..."):
            df = get_data_from_sheet(sheet_id, sheet_name)
            st.session_state.df = df
            if df is not None:
                st.success("Successfully connected!")
                intro = {"role": "assistant", "content": {"type": "text",
                                                          "content": "Hello! I'm ready to analyze your data. How can I help?"}}
                st.session_state.messages = [intro]
                st.session_state.df_schema = df.dtypes.to_string()
            else:
                st.error("Failed to load data.")

if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if content["type"] == "plot":
                st.pyplot(content["content"])
            elif content["type"] == "error":
                st.error(content["content"])
            else:
                st.markdown(content["content"])

    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": {"type": "text", "content": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                intent = classify_intent_by_rules(prompt)
                response = {}

                if intent == "greeting":
                    response = {"type": "text", "content": "Hello there! How can I help you with your data today?"}

                elif intent == "help_request":
                    # Using a clean, simple markdown string
                    response_text = """
Of course! I can help you in a few ways:
- **Answer Questions:** Ask me to calculate or find things in your data (e.g., *'What is the total Revenue?'*).
- **Create Charts:** Ask me to plot your data (e.g., *'Create a bar chart of sales by category'*).
- **Get Ideas:** Ask me to *'give me some sample questions'* if you're not sure where to start.
"""
                    response = {"type": "text", "content": response_text}

                elif intent == "query_suggestion":
                    suggestions = generate_query_suggestions(st.session_state.df_schema, prompt)
                    response_text = "Certainly! Here are a few questions you could ask:\n\n" + "\n".join(
                        [f"- *{s}*" for s in suggestions])
                    response = {"type": "text", "content": response_text}

                elif intent == "data_query":
                    code = generate_python_code(prompt, st.session_state.df_schema)
                    if code and is_valid_code(code):
                        response = execute_code(code, df)
                    else:
                        logger.warning(f"AI generated invalid or no code for prompt: '{prompt}'")
                        response = {"type": "error",
                                    "content": "I wasn't able to generate a valid analysis for that question. Could you please try rephrasing it? For help, just ask 'what can you do?'."}

                # Render the response
                if response:
                    try:
                        if response["type"] == "plot":
                            st.pyplot(response["content"])
                        elif response["type"] == "error":
                            st.error(response["content"])
                        else:
                            st.markdown(response["content"])
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        logger.error(f"Failed to render response: {e}", exc_info=True)
                        st.error("I encountered an issue displaying the response. Please check the logs.")
                else:
                    st.error("Sorry, I couldn't process that request.")
else:
    st.info("👋 Welcome! Please enter your Google Sheet details in the sidebar to get started.")
