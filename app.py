import os
import subprocess
import json
import datetime
import sqlite3
import re
import glob
import math
from openai import OpenAI
import platform
from PIL import Image
from io import BytesIO
import pytesseract
import calendar
from googletrans import Translator
import base64
import langdetect
from google.cloud import speech
from pydub import AudioSegment
import io
from dotenv import load_dotenv
from dateutil.parser import parse
from flask import Flask, request, jsonify, Response
import numpy as np
import asyncio
import requests

load_dotenv()

# Set up directories relative to this script's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Set your OpenAI API key from the environment variable (AIPROXY_TOKEN)
# openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"  # ✅ Correct API base
# openai.api_key = os.getenv("AIPROXY_TOKEN")
client = OpenAI(
    api_key=os.getenv("AIPROXY_TOKEN"),  # Ensure this is set in your .env file
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"  # ✅ Correct AI Proxy URL
)


# ======================
# Dummy Helper Functions (Replace these with actual implementations if available)
# ======================

def dummy_llm_extract_email(content):
    match = re.search(r'[\w\.-]+@[\w\.-]+', content)
    return match.group(0) if match else ""

def validate_path(path):
    # Resolve absolute path and ensure it starts with the DATA_DIR
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(os.path.abspath(DATA_DIR)):
        raise PermissionError(f"Access to {abs_path} is not allowed.")
    return abs_path

# ======================
# LLM Task Parsing Function
# ======================
def parse_task_with_llm(task_description):
    """
    Use GPT-4 (via OpenAI API Proxy) to parse the task description
    and output a JSON object with an "operations" array.
    If the task description is not in English, we attempt to translate it first.
    """
    try:
        # Detect the language of the task description
        detected_lang = langdetect.detect(task_description)
        print(f"Detected Language: {detected_lang}")
        translated_task = task_description 
        # Translate to English if necessary (you can integrate with a translation API or process it differently)
        if detected_lang != 'en':
            print(f"Translating from {detected_lang} to English...")
            translator = Translator()
            translated_task  = asyncio.run(translator.translate(task_description, src=detected_lang, dest='en')).text
            print(f"Translated Task Description: {translated_task}")
            # You can use Google Translate API, DeepL, or any translation service
            # For now, we simply pass the text as is.
            # In production, you might want to add a translation service here.
        prompt = f"""
You are an automation agent that understands and executes tasks described in natural language.
Under no circumstances should any operation delete or remove any files or data from the system. If the task description asks for deletion, simply ignore that request or return an error.
Given the following plain-English task description,
determine which operations need to be performed. Output a JSON object with one field "operations",
which is an array of operation objects. Each operation object must have an "operation" field that is one of:
"format_file", "count_dates", "sort_contacts", "extract_logs", "index_docs", "extract_email",
"extract_credit_card", "find_similar_comments", "query_tickets", "fetch_api", "clone_git", 
"run_sql_query", "scrape_website", "resize_image", "transcribe_audio", "md_to_html".
If the task description mentions ticket sales, ensure that an operation 'query_tickets' is returned with the specific ticket type.
If it involves fetching data from an API, cloning a repo, running SQL queries, scraping a website, image processing, audio transcription, or Markdown conversion, return the appropriate operation.
If the task involves fetching data from an API, the returned JSON object must include both the api_url and output_filename keys.
If it involves cloning a git repo, include repo_url and commit_message.
If running an SQL query, include db_filename and query.
If scraping a website, include url and output_filename.
If resizing an image, include input_image_filename, output_image_filename, and optionally a size parameter.
If transcribing audio, include input_audio_filename and output_text_filename.
If converting Markdown to HTML, include input_md_filename and output_html_filename.
Task Description: "{translated_task}"
Respond with only the JSON object.
"""
        # Use OpenAI to parse the task description
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # ✅ Use supported models
            messages=[
                {"role": "system", "content": "You are a helpful automation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0,
        )
        
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()  # Strip leading ```json and trailing ```
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()  # General case for ``` ... ```
        
        print("🔹 Cleaned LLM response:", result_text)  # Debugging output
        
        parsed = json.loads(result_text)
        return parsed.get("operations", []), translated_task
    except langdetect.lang_detect_exception.LangDetectException as e:
        print(f"Language detection error: {e}")
        return [], task_description  # Fallback if language detection fails
    except Exception as e:
        print("LLM parsing error:", e)
        return [], task_description  # Return empty operations if LLM parsing fails

# ======================
# Task Handlers (A2 to A10)
# ======================

def handle_format_file():
    filepath = validate_path(os.path.join(DATA_DIR, "format.md"))
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
    # Determine the correct command for npx on Windows vs. other OSes.
    npx_cmd = "npx.cmd" if platform.system() == "Windows" else "npx"
    cmd = [npx_cmd, "prettier@3.4.2", "--write", filepath]
    try:
        # Run the command and capture output and error
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise  # Reraise the exception to notify the caller
    return "Operation 'format_file' completed: format.md formatted using prettier@3.4.2."

def handle_count_dates(day_name):
    day_name = day_name.capitalize()  # Normalize input to proper case
    if day_name not in calendar.day_name:
        return jsonify({"error": f"Invalid day name: {day_name}. Please provide a valid day of the week."}), 400

    # Convert the day name to weekday number (0=Monday, 1=Tuesday, ..., 6=Sunday)
    target_weekday = list(calendar.day_name).index(day_name)
    # Count the number of Wednesdays in /data/dates.txt and write to /data/dates-wednesdays.txt
    input_file = validate_path(os.path.join(DATA_DIR, "dates.txt"))
    output_file = validate_path(os.path.join(DATA_DIR, f"dates-{day_name.lower()}.txt"))

    print(f"📁 Checking file: {input_file}")

    # 🛑 Check if file exists
    if not os.path.exists(input_file):
        from flask import abort
        abort(404, description="dates.txt file not found.")

    count = 0  # Initialize Wednesday counter

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            date_str = line.strip()
            if not date_str:
                continue  # Skip empty lines

            try:
                # ✅ Smart date parsing (handles any format)
                dt = parse(date_str, fuzzy=True)

                # ✅ Check if it's a Wednesday (0=Monday, 1=Tuesday, 2=Wednesday)
                if dt.weekday() == target_weekday:
                    count += 1
            except Exception:
                print(f"Skipping invalid date: {date_str}")  # Debugging

        print(f"✅ Found {count} {day_name}s")   # Log result

    # ✅ Write result to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(count))

    return jsonify({
        "message": f"Found {count} Wednesdays",
        "output": str(count),  # Make sure this is a string
        "output_file": output_file
    })


def handle_sort_contacts():
    # Sort /data/contacts.json by last_name then first_name and write to /data/contacts-sorted.json
    input_path = validate_path(os.path.join(DATA_DIR, "contacts.json"))
    output_path = validate_path(os.path.join(DATA_DIR, "contacts-sorted.json"))
    with open(input_path, "r", encoding="utf-8") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_contacts, f, indent=2)
    return "Operation 'sort_contacts' completed: contacts sorted."

def handle_extract_logs():
    # Extract the first line of the 10 most recent .log files from /data/logs/ and write to /data/logs-recent.txt
    logs_dir = validate_path(os.path.join(DATA_DIR, "logs"))
    output_path = validate_path(os.path.join(DATA_DIR, "logs-recent.txt"))
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"{logs_dir} not found")
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    recent_logs = log_files[:10]
    lines = []
    for log in recent_logs:
        with open(log, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            lines.append(first_line)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return "Operation 'extract_logs' completed: processed recent log files."

def handle_index_docs():
    # For every Markdown file in /data/docs/, extract its first H1 and create an index JSON mapping.
    docs_dir = validate_path(os.path.join(DATA_DIR, "docs"))
    index = {}
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, docs_dir)
                title = None
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.lstrip().startswith("#"):
                            title = line.lstrip("#").strip()
                            break
                if title:
                    index[rel_path] = title
    output_path = validate_path(os.path.join(docs_dir, "index.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return "Operation 'index_docs' completed: Markdown docs indexed."

def handle_extract_email():
    # Extract the sender email from /data/email.txt using LLM (dummy extraction here)
    input_path = validate_path(os.path.join(DATA_DIR, "email.txt"))
    output_path = validate_path(os.path.join(DATA_DIR, "email-sender.txt"))
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    sender_email = dummy_llm_extract_email(content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sender_email)
    return f"Operation 'extract_email' completed: extracted sender email {sender_email}."

def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # Save the image in PNG format
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def handle_extract_credit_card():
    # Extract the credit card number from /data/credit-card.png using LLM (dummy extraction here)
    input_path = validate_path(os.path.join(DATA_DIR, "credit_card.png"))
    output_path = validate_path(os.path.join(DATA_DIR, "credit_card.txt"))

    # img_path = r'C:\Users\Shahzade Alam\Desktop\TDS_P1\data\credit_card.png'
    # img_base64 = encode_image_to_base64(input_path)
    # print("🔹 Image encoded to base64.")

    if os.path.exists(input_path):
        extracted_text = pytesseract.image_to_string(input_path)
        card_number = ''.join(filter(str.isdigit, extracted_text)) 
        if card_number:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(card_number)
            return f"Operation 'extract_credit_card' completed: extracted credit card number {card_number}."
        else:
            raise ValueError("No valid card number extracted.")
        
        # return f"Operation 'extract_credit_card' completed: extracted credit card number {card_number}."
        # Assuming your API supports this method, you'll send the image as base64-encoded string
    #     response = client.chat.completions.create(
    #     model="gpt-4o-mini",  # or the model name for image processing
    #     messages=[
    #         {"role": "user", "content": "Can you analyze this image?"}
    #     ],
    #     image=[{
    #         "base64": img_base64,
    #     }]
    # )
    else:
        print(f"File not found at {input_path}")

def get_embedding(text: str) -> np.ndarray:
    """Fetch embedding from OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def handle_find_similar_comments():
    input_path = validate_path(os.path.join(DATA_DIR, "comments.txt"))
    output_path = validate_path(os.path.join(DATA_DIR, "comments-similar.txt"))
    # Find the most similar pair of comments in /data/comments.txt using embeddings.
    try:
        # Read comments from file
        with open(input_path, "r", encoding="utf-8") as f:
            comments = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(comments) < 2:
            return {"error": "Not enough comments to find a similar pair."}
        
        # Compute embeddings
        embeddings = {comment: get_embedding(comment) for comment in comments}

        # Find the most similar pair
        max_similarity = -1
        most_similar_pair = ("", "")
        
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                sim = cosine_similarity(embeddings[comments[i]], embeddings[comments[j]])
                if sim > max_similarity:
                    max_similarity = sim
                    most_similar_pair = (comments[i], comments[j])

        # Write the most similar pair to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{most_similar_pair[0]}\n{most_similar_pair[1]}\n")
        
        return "Operation 'find_similar_comments' completed: similar comments found."

    except Exception as e:
        return {"error": str(e)}

def handle_query_tickets(ticket_type):
    # Build the path for the SQLite database file.
    db_path = validate_path(os.path.join(DATA_DIR, "ticket-sales.db"))
    output_path = validate_path(os.path.join(DATA_DIR, f"ticket-sales-{ticket_type.lower()}.txt"))
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"{db_path} not found")
    
    # Connect to the SQLite database and execute the query.
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = "SELECT SUM(units * price) FROM tickets WHERE type = ?"
    cur.execute(query, (ticket_type,))
    result = cur.fetchone()[0]
    conn.close()
    
    # Write the result to the corresponding file.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(result))
    
    return f"Operation 'query_tickets' completed: total {ticket_type} ticket sales = {result}."

def handle_fetch_api(api_url, output_filename):
    # Validate the output path to ensure it's within /data
    output_path = validate_path(os.path.join(DATA_DIR, output_filename))
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an error if the request failed
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return f"Operation 'fetch_api' completed: data fetched from {api_url} and saved to {output_filename}."

def handle_clone_git(repo_url, commit_message):
    # Clone the repo into a subdirectory of /data (e.g., /data/git_repos/<repo_name>)
    repo_name = repo_url.rstrip('/').split('/')[-1]
    clone_dir = validate_path(os.path.join(DATA_DIR, "git_repos", repo_name))
    os.makedirs(clone_dir, exist_ok=True)
    try:
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
        # Make a dummy change to commit (for example, create a file)
        dummy_file = os.path.join(clone_dir, "dummy.txt")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("This is a dummy commit.")
        subprocess.run(["git", "-C", clone_dir, "add", "dummy.txt"], check=True)
        subprocess.run(["git", "-C", clone_dir, "commit", "-m", commit_message], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git operation failed: {e}")
    return f"Operation 'clone_git' completed: repo cloned and commit made with message '{commit_message}'."

def handle_run_sql_query(db_filename, query):
    # Ensure that the database file is within /data
    db_path = validate_path(os.path.join(DATA_DIR, db_filename))
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"{db_path} not found")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.commit()
    conn.close()
    # Optionally, you can save the result to a file within /data
    result_file = validate_path(os.path.join(DATA_DIR, "sql_query_result.txt"))
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(str(result))
    return f"Operation 'run_sql_query' completed: query executed and result saved."

from bs4 import BeautifulSoup  # Make sure to install beautifulsoup4

def handle_scrape_website(url, output_filename):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/90.0.4430.93 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # For simplicity, extract all text from the page
    page_text = soup.get_text(separator="\n")
    output_path = validate_path(os.path.join(DATA_DIR, output_filename))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page_text)
    return f"Operation 'scrape_website' completed: data from {url} saved to {output_filename}."

def handle_resize_image(input_image_filename, output_image_filename, max_width=None, max_height=None, quality=50, size=None):
    # Construct the full path for the input file
    input_path = validate_path(os.path.join(DATA_DIR, input_image_filename))
    
    # Verify that the input file exists
    if not os.path.exists(input_path):
        return f"Error: Input file {input_image_filename} not found."

    # Open the image
    image = Image.open(input_path)

    # Resize if dimensions are provided
    if max_width and max_height:
        image.thumbnail((max_width, max_height))  # Maintains aspect ratio

    # Construct the full path for the output file
    output_path = os.path.join(DATA_DIR, output_image_filename)

    # Compress and save the image
    image.save(output_path, format="JPEG", quality=quality)

    return f"Operation 'resize_image' completed: Saved to {output_image_filename}."

def handle_transcribe_audio(input_audio_filename, output_text_filename):
    # Validate input and output paths.
    input_path = validate_path(os.path.join(DATA_DIR, input_audio_filename))
    output_path = validate_path(os.path.join(DATA_DIR, output_text_filename))
    # Dummy transcription – in reality, integrate with a transcription service.
    transcription = "This is a dummy transcription of the audio."
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    return f"Operation 'transcribe_audio' completed: transcription saved to {output_text_filename}."

import markdown  # Make sure to install the markdown library

def handle_md_to_html(input_md_filename, output_html_filename):
    input_path = validate_path(os.path.join(DATA_DIR, input_md_filename))
    output_path = validate_path(os.path.join(DATA_DIR, output_html_filename))
    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html = markdown.markdown(md_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return f"Operation 'md_to_html' completed: HTML generated and saved to {output_html_filename}."

import csv

# ======================
# Mapping from Operation Name to Handler Function
# ======================
operation_map = {
    "format_file": handle_format_file,
    "count_dates": handle_count_dates,
    "sort_contacts": handle_sort_contacts,
    "extract_logs": handle_extract_logs,
    "index_docs": handle_index_docs,
    "extract_email": handle_extract_email,
    "extract_credit_card": handle_extract_credit_card,
    "find_similar_comments": handle_find_similar_comments,
    "query_tickets": handle_query_tickets,
    "fetch_api": handle_fetch_api,
    "clone_git": handle_clone_git,
    "run_sql_query": handle_run_sql_query,
    "scrape_website": handle_scrape_website,
    "resize_image": handle_resize_image,
    "transcribe_audio": handle_transcribe_audio,
    "md_to_html": handle_md_to_html,
}

# ======================
# Dispatcher that Uses the LLM's Plan to Execute Operations
# ======================
def execute_operation(task_description):
    # Use the LLM to get a structured plan
    operations, translated_task = parse_task_with_llm(task_description)
    print("LLM returned operations:", operations)
    
    # If no operations are returned, optionally use a fallback heuristic.
    if not operations:
        # Fallback: simple keyword matching (you can extend this as needed)
        desc = translated_task.lower()
        if "ticket" in desc and "sales" in desc:
            operations = [{"operation": "query_tickets"}]
        elif "format" in desc and "format.md" in desc:
            operations = [{"operation": "format_file"}]
        elif "dates.txt" in desc and "wednesday" in desc:
            operations = [{"operation": "count_dates"}]
        # Add other heuristics as desired.
        elif "contacts.json" in desc and "sort" in desc:
            operations = [{"operation": "sort_contacts"}]
    
    results = []
    for op in operations:
        op_name = op.get("operation")

        if "delete" in op_name.lower():
            results.append("Error: Data deletion is not permitted.")
            continue

        if op_name == "count_dates":
            # Extract the specific day name (e.g., "Monday", "Wednesday", etc.) from the task description
            day_name_match = re.search(r"(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)", translated_task, re.IGNORECASE)
            if day_name_match:
                day_name = day_name_match.group(0)
                res = handle_count_dates(day_name)
                results.append(res)
            else:
                results.append("Error: No valid day of the week found in the task description.")

        elif op_name == "query_tickets":
            # Search for ticket types (Gold, Silver, Bronze) in the task description.
            ticket_type_match = re.search(r"(Gold|Silver|Bronze)", translated_task, re.IGNORECASE)
            if ticket_type_match:
                ticket_type = ticket_type_match.group(0).capitalize()
            else:
                ticket_type = "Gold"  # Default if none is specified.
            res = handle_query_tickets(ticket_type)
            results.append(res)

        elif op_name in operation_map:
            try:
                func = operation_map[op_name]
        # Remove the "operation" key to leave just the parameters.
                params = {k: v for k, v in op.items() if k != "operation"}
        # Call the handler with parameters if any exist.
                res = func(**params) if params else func()
                results.append(res)
            except Exception as e:
                results.append(f"Error executing {op_name}: {e}")
        else:
            results.append(f"Operation '{op_name}' is not supported.")
    
    # Return all operation results as a single string.
    return "\n".join(results)

# ======================
# Flask API Endpoints
# ======================
app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run_task():
    task = request.args.get("task")
    if not task:
        return jsonify({"error": "No task provided"}), 400
    try:
        result = execute_operation(task)
        return jsonify({"result": result}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Agent error: " + str(e)}), 500

@app.route("/read", methods=["GET"])
def read_file():
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No file path provided"}), 400
    file_path = file_path.strip()  # Remove stray whitespace/newlines
    absolute_path = os.path.normpath(os.path.join(DATA_DIR, file_path))
    # Security check: ensure absolute_path is within DATA_DIR.
    if not absolute_path.startswith(os.path.abspath(DATA_DIR)):
        return jsonify({"error": "Access to this file is not allowed"}), 403
    if not os.path.exists(absolute_path):
        return "", 404
    try:
        with open(absolute_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content, mimetype="text/plain"), 200
    except Exception as e:
        return jsonify({"error": "Error reading file: " + str(e)}), 500
    
@app.route("/filter_csv", methods=["GET"])
def filter_csv():
    # Expect parameters: path (CSV file relative to /data) and filter conditions (e.g., column and value)
    file_path = request.args.get("path")
    filter_column = request.args.get("column")
    filter_value = request.args.get("value")
    if not file_path or not filter_column or not filter_value:
        return jsonify({"error": "Missing required parameters."}), 400
    absolute_path = validate_path(os.path.join(DATA_DIR, file_path))
    if not os.path.exists(absolute_path):
        return jsonify({"error": "CSV file not found."}), 404

    filtered_rows = []
    with open(absolute_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get(filter_column) == filter_value:
                filtered_rows.append(row)
    return jsonify(filtered_rows), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
