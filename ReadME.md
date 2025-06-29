
### Part 1: The Master Prompt for Future Sessions

*(Copy and paste the entire text below into a new chat window with me in the future to instantly bring me up to speed.)*

**[START OF MASTER PROMPT]**

**Project Title:** Conversational AI Analyst for Google Sheets

**My Role:** You are my expert AI development partner. You have full context on the project detailed below. Your goal is to help me understand, modify, and extend this project based on the final, production-ready architecture we developed. Do not deviate from this established architecture unless I explicitly ask you to.

**Project Goal:** To create a web application using Streamlit that allows a user to connect to complex, real-world Google Sheets and have a robust, intelligent conversation about their data. The application must reliably handle greetings, generate charts, and answer questions about the data, even when the source spreadsheet has messy, multi-level, or merged-cell headers.

**Evolutionary Journey & Key Decisions (This is our shared memory):**

This project's success came from a rigorous, iterative process of identifying and solving critical failures. This journey is essential to understanding the final design choices.

1.  **Initial State:** A simple script with a single LLM prompt. This failed immediately.
2.  **Problem: Unreliable Code and Intent.** The initial LLM was hallucinating code, misinterpreting questions, and failing on simple tasks.
    *   **Solution: The Agentic (Tool-Based) Architecture.** We abandoned the single-prompt approach and re-architected the application into a proper AI agent. This involved  creating a "Conductor" LLM responsible for choosing the right "tool" (a Python function) for the job, such as `generate_chart_code` or `answer_general_question`.

3.  **Problem: "Creative" Conductor and Tool Hallucination.** The Conductor LLM was "too creative" and would ignore the list of valid tools, inventing its own tool names (`gpt-2`, `DataProfiler`) and causing the system to crash.
    *   **Solution: The "No Escape" Conductor Prompt.** We radically simplified and hardened the Conductor's system prompt. We gave it a simple list of tools and a non-negotiable rule: **"If you are unsure, you MUST choose the 'clarify_or_answer_general' tool."** This constrained its creativity and forced it to be an obedient router.

4.  **Problem: Catastrophic Data Loading Failures.** This was the most critical and persistent bug. The data loader repeatedly failed on real-world sheets with multi-line, merged-cell headers and duplicate sub-headers. Our attempts to use "magic" heuristic detection or basic pandas functions led to a cascade of errors (`Length mismatch`, `Duplicate column`, `TypeError: int64 not JSON serializable`).
    *   **Solution: The Definitive ETL Data Pipeline.** We completely re-architected the data loading process into a professional ETL (Extract, Transform, Load) pipeline. The final, robust `load_and_restructure_data` function now follows a precise, industry-standard method:
        a.  It reads the raw data into a basic DataFrame.
        b.  It gives the **user full control** via a "Header Rows" input in the UI. This was a critical design choice to move from a brittle "magic" solution to a robust, professional one.
        c.  It isolates the specified header rows and uses a **horizontal forward-fill (`.ffill(axis=1)`)** to correctly handle merged cells.
        d.  It **manually combines** the filled header rows into a single, clean list of headers, explicitly following the user's logic (`BuildName_MetricName`).
        e.  It correctly filters out ignored columns and de-duplicates the final combined headers before creating the final DataFrame. This architecture is now robust enough to handle any complex spreadsheet layout.

5.  **Problem: Unhelpful, Repetitive Responses.** The agent would get stuck in loops, providing the same generic suggestions when asked for "more" or misunderstanding nuanced requests like "give me chart suggestions with chart names."
    *   **Solution: A Specialized Toolbox.** We created a more diverse set of tools for the Conductor to choose from, including `list_chart_types` (deterministic answer), `generate_overall_insights` (proactive analysis), and an improved `generate_data_specific_suggestions` that uses a non-zero temperature to provide varied responses on each call.

**Final Core Architecture:**

*   **Frontend:** Streamlit (`app.py`).
*   **Data Loading:** A robust, user-controlled **ETL Pipeline** (`load_and_restructure_data`) that correctly parses complex, multi-level, and merged-cell headers into a clean, flat DataFrame with descriptive, combined column names.
*   **Agent Brain (The Conductor):** A function (`run_conductor`) powered by a strict, "No Escape" system prompt. Its only job is to receive the user query and chat history, and then choose the single most appropriate tool from its toolbox.
*   **Toolbox (The Agent's Capabilities):**
    *   `generate_chart_code`: A code-generation tool with a hardened "Chain of Thought" prompt to produce reliable Python code for charts.
    *   `generate_data_query_code`: A similar tool for generating code to  answer precise numerical questions.
    *   `list_chart_types`: A deterministic function that returns a hardcoded list of chart capabilities.
    *   `generate_data_specific_suggestions`: A tool that uses a non-zero temperature to provide varied, data-aware example questions.
    *   `clarify_or_answer_general`: A safe fallback tool for greetings and ambiguous queries.
*   **Execution & Safety:** A `safe_call_tool` function that acts as a safety net, preventing crashes from LLM parameter hallucinations, and an `execute_code_and_capture_output` function to safely run the generated Python code.

**[END OF MASTER PROMPT]**