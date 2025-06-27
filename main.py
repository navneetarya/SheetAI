import os
import io
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
CREDENTIALS_FILE = 'credentials.json'


# --- FUNCTION 1: GOOGLE SHEETS AUTHENTICATION & DATA FETCHING ---
def get_data_from_sheet(spreadsheet_id, range_name):
    # This function is unchanged
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])

        if not values:
            print('No data found in the sheet.')
            return None
        else:
            return pd.DataFrame(values[1:], columns=values[0])
    except Exception as e:
        print(f"An error occurred while fetching data from Google Sheets: {e}")
        return None


# --- FUNCTION 2: AI CODE GENERATION (USING GROQ) ---
def generate_python_code(query, df_head):
    """
    Sends the user's query and data schema to Groq to generate Python code.
    """
    try:
        client = Groq()
    except Exception as e:
        print("Error initializing Groq client. Is the GROQ_API_KEY environment variable set correctly?")
        print(e)
        return ""

    prompt = f"""
    You are a data analyst expert. You are given a pandas DataFrame named `df`.
    The first 5 rows of the DataFrame look like this:
    {df_head}

    Your task is to write a short Python script to answer the following user query.
    User Query: "{query}"

    Instructions:
    1. The DataFrame is already loaded as `df`. Do NOT include code to load the data.
    2. Your script should print the final answer to the console.
    3. If the user asks for a plot or chart, use the `matplotlib.pyplot` library. The library is already imported as `plt`.
       - After creating the plot, use `plt.show()` to display it.
    4. Your output should ONLY be the Python code. Do not include any explanations, comments, or markdown formatting.
    """

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst that generates Python code."},
                {"role": "user", "content": prompt}
            ]
        )
        code = response.choices[0].message.content

        # --- NEW, MORE ROBUST CLEANING LOGIC ---
        # 1. Strip leading/trailing whitespace
        code = code.strip()
        # 2. Check for and remove markdown blocks (e.g., ```python ... ``` or ``` ... ```)
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        # 3. Strip again to be safe
        return code.strip()
        # --- END OF NEW LOGIC ---

    except Exception as e:
        print(f"An error occurred while communicating with Groq API: {e}")
        return ""


# --- FUNCTION 3: CODE EXECUTION ---
def execute_generated_code(code, df):
    # This function is unchanged
    if not code:
        print("No code was generated, skipping execution.")
        return

    print("\n--- Executing Code ---")
    print(code)
    print("----------------------\n")

    output_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_capture

    try:
        exec(code, {'df': df, 'pd': pd, 'plt': plt})
    except Exception as e:
        sys.stdout = original_stdout
        print(f"An error occurred while executing the generated code: {e}")
    finally:
        sys.stdout = original_stdout

    result = output_capture.getvalue()
    print("--- Result ---")
    if result:
        print(result)
    else:
        print("(No text output was produced. If you asked for a chart, it should have appeared in a new window.)")
    print("--------------\n")


# --- FUNCTION 4: MAIN APPLICATION LOOP ---
def main():
    # This function is unchanged
    print("Welcome to the Google Sheets AI Analyst!")

    spreadsheet_id = input("Please enter your Google Sheet ID: ")
    sheet_name = input("Please enter the name of the sheet (e.g., Sheet1): ")

    df = get_data_from_sheet(spreadsheet_id, sheet_name)

    if df is not None:
        print("\nSuccessfully loaded data. Here are the first 5 rows:")
        print(df.head())

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        while True:
            query = input("Ask a question about your data (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            generated_code = generate_python_code(query, df.head())
            execute_generated_code(generated_code, df)


# This line tells Python to start by running the main() function.
if __name__ == '__main__':
    main()