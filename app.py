import streamlit as st
import pandas as pd
import os
import io
import re
import warnings
import json
import inspect
from contextlib import redirect_stdout
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from groq import Groq
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import sys

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

st.set_page_config(page_title="Conversational Sheet Analyst", layout="wide")
st.title("🤖 Structure-Aware Analyst Agent")

# --- Google Authentication & Data Loading (Cached) ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']


@st.cache_resource
def authenticate_google():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


@st.cache_data(ttl=600)
def get_sheet_names(_creds, spreadsheet_id):
    try:
        service = build('sheets', 'v4', credentials=_creds)
        sheets_meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        return [sheet['properties']['title'] for sheet in sheets_meta['sheets']]
    except Exception as e:
        st.error(f"Error fetching sheet names: {e}")
        return []


# DEFINITIVE FIX: New data loader with robust de-duplication
@st.cache_data(ttl=600)
def load_and_restructure_data(_creds, spreadsheet_id, sheet_name, header_rows_str="0,1"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            header_rows = [int(x.strip()) for x in header_rows_str.split(',')]

            service = build('sheets', 'v4', credentials=_creds)
            result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
            values = result.get('values', [])

            if not values: return None

            raw_df = pd.DataFrame(values).replace('', pd.NA)

            # --- Manual Header Reconstruction ---
            top_header = raw_df.iloc[header_rows[0]].ffill()
            bottom_header = raw_df.iloc[header_rows[1]]

            final_headers = []
            for i, (top, bottom) in enumerate(zip(top_header, bottom_header)):
                if pd.notna(top) and pd.notna(bottom) and top != bottom:
                    final_headers.append(f"{top}_{bottom}")
                else:
                    final_headers.append(bottom if pd.notna(bottom) else top)

            # --- Post-Hoc De-duplication ---
            cols = pd.Series(final_headers)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else f"{dup}" for i in
                                                                 range(sum(cols == dup))]

            raw_df.columns = cols

            # --- DataFrame Creation and Cleanup ---
            data_start_row = max(header_rows) + 1
            df = raw_df.iloc[data_start_row:]

            df = df.set_index(df.columns[0])
            df.index.name = "End Point"

            df = df.apply(pd.to_numeric, errors='coerce').convert_dtypes()

            return df
        except Exception as e:
            st.error(f"Failed to load and restructure data. Error: {e}")
            logging.error(f"Data restructuring failed: {e}")
            return None


# --- LLM & Agent Tools (All functions from here down are stable) ---
def call_llm(system_prompt, user_prompt, model="llama3-70b-8192", json_mode=False, temperature=0.0):
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if json_mode:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, response_format={"type": "json_object"}
            )
        else:
            response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLM Error: {e}")
        return json.dumps({"error": f"LLM communication failed: {e}"})


def clean_code(code):
    if "```python" in code: code = code.split("```python", 1)[1]
    return code.strip().strip('`')


def list_chart_types():
    chart_list = ["Bar Chart", "Line Plot", "Scatter Plot", "Pie Chart", "Histogram"]
    formatted_list = "I can create the following types of charts for you:\n* " + "\n* ".join(chart_list)
    return json.dumps({"answer": formatted_list})


def generate_chart_code(prompt: str, schema: str):
    system_prompt = """You are an expert Python data visualization generator. Your response must be ONLY the Python code, wrapped in a markdown block.
STRICT RULES:
1. The pandas DataFrame is ALWAYS named `df`. The columns have descriptive, combined names like 'BuildName_MetricName'.
2. The code must generate a SINGLE chart. Use `fig, ax = plt.subplots()`.
3. Set a clear title and labels. DO NOT use `plt.show()`.
"""
    return call_llm(system_prompt, f"User prompt: {prompt}\nDataFrame schema:\n{schema}", json_mode=False)


def generate_data_query_code(prompt: str, schema: str):
    system_prompt = """You are an expert Python data analyst. Your response must be ONLY the Python code, wrapped in a markdown block.
STRICT RULES:
1. The pandas DataFrame is ALWAYS named `df`. The columns have descriptive, combined names like 'BuildName_MetricName'.
2. The code must PRINT the final answer as a full, human-readable sentence.
"""
    return call_llm(system_prompt, f"User prompt: {prompt}\nDataFrame schema:\n{schema}", json_mode=False)


def clarify_or_answer_general(prompt: str):
    system_prompt = """You are a helpful AI assistant. Your response must be a JSON object in the format: {"answer": "your response"}."""
    return call_llm(system_prompt, prompt, json_mode=True)


def execute_code_and_capture_output(code: str, df: pd.DataFrame):
    if not code or not isinstance(code, str): return {"type": "error", "content": "No code was generated."}
    clean = clean_code(code)
    if not clean: return {"type": "error", "content": "The generated code was empty."}
    logging.info(f"\n--- EXECUTING CLEANED CODE ---\n{clean}\n------------------------------")
    local_scope = {'df': df, 'pd': pd, 'plt': plt, 'mdates': mdates}
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(clean, local_scope)
        fig = local_scope.get('fig', plt.gcf())
        if fig and len(fig.get_axes()) > 0: return {"type": "plot", "content": fig}
        printed_output = buffer.getvalue()
        if printed_output: return {"type": "text", "content": printed_output}
        return {"type": "text", "content": "The code ran successfully but produced no output."}
    except Exception as e:
        logging.error(f"Code Execution Failed: {e}\nCode:\n{clean}")
        return {"type": "error", "content": f"I tried to run some code, but it failed: {e}"}


def run_conductor(prompt, chat_history, schema):
    tools_list = """
1.  `list_chart_types`: Use ONLY when the user asks a direct question about your chart types.
2.  `generate_chart_code`: Use when the user asks to create a specific chart or plot.
3.  `generate_data_query_code`: Use when the user asks for a specific number or calculation.
4.  `clarify_or_answer_general`: Fallback for greetings, vague follow-ups, or general questions.
"""
    system_prompt = f"""You are a "Conductor" AI. Your only job is to select the single most appropriate tool from the provided list to handle the user's request.
Your response MUST be a single, valid JSON object with "tool" and "parameters" keys.
**Available Tools:**\n{tools_list}
**RULES:**
- For tools needing the user's prompt, the value MUST be the user's entire, original prompt.
- For tools without parameters, "parameters" must be an empty object: {{}}.
- You MUST choose a tool from the list. If unsure, choose 'clarify_or_answer_general'.
"""
    history_str = "\n".join([f"{msg['role']}: {msg.get('content', '')}" for msg in chat_history])
    user_prompt_for_conductor = f"Chat History:\n{history_str}\n\nUser's Latest Prompt: \"{prompt}\"\n\nYour JSON decision:"
    return call_llm(system_prompt, user_prompt_for_conductor, json_mode=True)


TOOLS = {
    "list_chart_types": list_chart_types,
    "generate_chart_code": generate_chart_code,
    "generate_data_query_code": generate_data_query_code,
    "clarify_or_answer_general": clarify_or_answer_general,
}


def safe_call_tool(tool_name, params, **kwargs):
    if tool_name not in TOOLS: raise ValueError(f"Unknown tool: {tool_name}")
    tool_func = TOOLS[tool_name]
    tool_spec = inspect.signature(tool_func)
    valid_params = {}
    for param_name in tool_spec.parameters:
        if param_name in params:
            valid_params[param_name] = params[param_name]
        elif param_name in kwargs:
            valid_params[param_name] = kwargs[param_name]
    return tool_func(**valid_params)


# --- Streamlit UI ---
st.sidebar.header("🔗 Connect to Google Sheet")
if 'creds' not in st.session_state: st.session_state.creds = authenticate_google()
sheet_id = st.sidebar.text_input("Enter Google Sheet ID")
if sheet_id:
    try:
        sheet_list = get_sheet_names(st.session_state.creds, sheet_id)
        if sheet_list:
            sheet_name = st.sidebar.selectbox("Select Sheet Tab", sheet_list)

            header_rows_str = st.sidebar.text_input(
                "Header Rows (e.g., 0,1)",
                value="0,1",
                help="Enter the row number(s) of the header, separated by commas. The first row is 0."
            )

            if st.sidebar.button("Load & Analyze Sheet"):
                with st.spinner("Loading and restructuring data..."):
                    df = load_and_restructure_data(st.session_state.creds, sheet_id, sheet_name, header_rows_str)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.messages = []
                        st.success("Sheet loaded! How can I help you analyze this structured data?")
                    else:
                        st.error("Could not load data. Check sheet format and header rows.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")

if "df" in st.session_state:
    df = st.session_state.df
    st.write("### Structured Data Preview");
    st.dataframe(df.head())
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            if msg.get('type') == 'plot':
                st.pyplot(msg['content'])
            else:
                st.markdown(msg.get('content', ''))

    if user_prompt := st.chat_input("Ask about your structured data..."):
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = None
                decision_str, result_str = "", ""
                try:
                    schema_buffer = io.StringIO()
                    df.info(buf=schema_buffer)
                    schema_info = schema_buffer.getvalue()

                    decision_str = run_conductor(user_prompt, st.session_state.messages[-5:], schema_info)
                    decision = json.loads(decision_str)
                    tool_name = decision.get("tool")
                    params = decision.get("parameters", {})
                    logging.info(f"Conductor chose tool: {tool_name} with params: {params}")

                    kwargs = {'schema': schema_info}

                    if tool_name in ["generate_chart_code", "generate_data_query_code"]:
                        code_str = safe_call_tool(tool_name, params, **kwargs)
                        response = execute_code_and_capture_output(code_str, df)
                    elif tool_name in ["list_chart_types", "clarify_or_answer_general"]:
                        result_str = safe_call_tool(tool_name, params, **kwargs)
                        response = {"type": "text", "content": json.loads(result_str).get("answer")}
                    else:
                        response = {"type": "error", "content": f"Error: The AI chose an unknown tool '{tool_name}'."}

                except json.JSONDecodeError as e:
                    failed_str = result_str if result_str and result_str != decision_str else decision_str
                    logging.error(f"JSON Decode Error: {e}. Raw response that failed: {failed_str}")
                    response = {"type": "error",
                                "content": "I had trouble processing that request. Please try rephrasing."}
                except Exception as e:
                    logging.error(f"An unexpected error occurred in the main loop: {e}")
                    response = {"type": "error", "content": f"An unexpected error occurred: {e}"}

                if response:
                    if response.get('type') == 'plot':
                        st.pyplot(response['content'])
                    elif response.get('type') == 'text':
                        st.markdown(response.get('content', ''))
                    elif response.get('type') == 'error':
                        st.error(response.get('content', ''))
                    st.session_state.messages.append({"role": "assistant", **response})
else:
    st.info("Please enter a Google Sheet ID in the sidebar to begin.")