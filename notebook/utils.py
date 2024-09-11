import pprint
from openai import OpenAI
import re
import json
import fitz
import os
import io

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
import base64
 
# Function to create the prompt
def create_prompt(question):
    question_prompt = f"Question:\n{question}\n"
   
    prompt_instruct_template = """
    Instructions:
    Provide an answer to the question.
    After the answer, include references used to generate the answer to the question in the following format:
    <ref>
    [
      {
        "file_name.extension": [
          {
            "text": "1st Exact content body of the reference in the original language.",
            "page_number": page_number_as_integer
          },
          {
            "text": "N-th Exact content body of the reference in the original language.",
            "page_number": page_number_as_integer
          }
        ]
      },
      {
        "file_name.extension": [
          {
            "text": "1st Exact content body of the reference in the original language.",
            "page_number": page_number_as_integer
          },
          {
            "text": "N-th Exact content body of the reference in the original language.",
            "page_number": page_number_as_integer
          }
        ]
      },
      ...
    ]
    <\ref>
 
    Only include references containing the keywords from the question.
    Group references by file, and include the page number for each reference.
    Use double quotations in the reference dictionaries and retain the original format of the references without optimization.
    Return the references as a list of dictionaries, each dictionary corresponding to a file name.
    """
    return question_prompt + prompt_instruct_template
 
# Function to get the AI response
def get_ai_response(client, assistant_id, thread_id, question):
    try:
        content = create_prompt(question)
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
 
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
 
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread_id
            )
            return messages.to_dict()
        else:
            print("Run not completed, status:", run.status)
            return None
 
    except Exception as e:
        print("Error in get_ai_response:", str(e))
        return None
 
# Function to extract and process the response and references
def process_response(messages):
    try:
        json_schema = messages["data"][0]
        response_text = json_schema["content"][0]["text"]["value"]
       
        # Split the response_text into response_ai and references_raw using regex
        response_ai, references_raw = re.split(r'<ref>|<\/ref>', response_text)[0:2]
 
        # Convert the cleaned references_raw into a dictionary
        references_dict = json.loads(references_raw)
       
        return response_ai.strip(), references_dict
 
    except Exception as e:
        print("Error in process_response:", str(e))
        return None, None
 
# Function to substitute text
def substitute_text(text, mapping):
    try:
        for key, value in mapping.items():
            text = text.replace(key, value)
        return text
    except Exception as e:
        print("Error in substitute_text:", str(e))
        return text
 
# Function to apply substitutions in references
def apply_substitutions(references_dict, substitution_mapping):
    try:
        for item in references_dict:
            for key, value in item.items():
                for entry in value:
                    entry['text'] = substitute_text(entry['text'], substitution_mapping)
        return references_dict
    except Exception as e:
        print("Error in apply_substitutions:", str(e))
        return references_dict
 
# Function to highlight text in the PDF
def incremental_search_and_highlight(pdf_path, search_data):
    try:
        doc = fitz.open(pdf_path)
        found_pieces_dict = {}
        file_name = os.path.basename(pdf_path)
       
        found_pieces = []
       
        for entry in search_data:
            text = entry['text']
            start_page_number = entry['page_number'] - 1  # Pages are zero-indexed in fitz
           
            start_idx = 0
            piece = ""
            current_page = start_page_number
           
            while start_idx < len(text):
                piece += text[start_idx]
               
                # If the next character is a newline, stop expanding and save the piece
                if start_idx + 1 < len(text) and text[start_idx + 1] == '\n':
                    if len(piece) > 5:
                        found_pieces.append((piece, current_page))
                    start_idx += 2  # Move past the newline
                    piece = ""
                    continue
 
                # Search on the current page and the next page
                page = doc.load_page(current_page)
                next_page = None
                if current_page + 1 < len(doc):
                    next_page = doc.load_page(current_page + 1)
               
                found_in_current_page = page.search_for(piece)
                found_in_next_page = next_page.search_for(piece) if next_page else None
 
                if found_in_current_page:
                    start_idx += 1
                elif found_in_next_page:
                    # If found in the next page, move to that page
                    current_page += 1
                    start_idx += 1
                else:
                    # If not found in either page, finalize the current piece
                    if piece[:-1] and len(piece[:-1]) > 5:
                        found_pieces.append((piece[:-1], current_page))
                    # Restart search from the current character
                    piece = text[start_idx]
                    start_idx += 1
 
            # Handle last piece if any
            if piece and len(piece) > 5:
                found_pieces.append((piece, current_page))
           
            # Highlight all found pieces on the relevant pages
            for piece, page_num in found_pieces:
                if len(piece) > 5:
                    page = doc.load_page(page_num)
                    instances = page.search_for(piece)
                    for inst in instances:
                        page.add_highlight_annot(inst)
       
        # Add found pieces to dictionary for this file
        found_pieces_dict[file_name] = found_pieces
       
        # Save the modified PDF to an in-memory buffer
        pdf_buffer = io.BytesIO()
        doc.save(pdf_buffer)
        doc.close()
       
        # Return the buffer contents
        pdf_buffer.seek(0)  # Reset buffer position to the beginning
        return found_pieces_dict, pdf_buffer.read()
 
    except Exception as e:
        print("Error in incremental_search_and_highlight:", str(e))
        return {}
 

 
@app.route('/app', methods=['POST'])
# Main function to orchestrate the workflow
def main():
    data = request.get_json()
    question = data["question"]
    try:
        client = OpenAI()
        assistant_id = "asst_fe4VWMpLT0W04Wpc8A8JQ2rg"
        thread_id = "thread_LIDxHch50hsIm7qT8iIXXXcm"
       
        # question = "trovami gli ordini di vendita di ACME del 2023 contenenti i prodotti 'levigatrice' o 'sega circolare'"
       
        messages = get_ai_response(client, assistant_id, thread_id, question)
        if messages:
            response_ai, references_dict = process_response(messages)
           
            if references_dict:
                substitution_mapping = {
                    'à': '`\na',
                }
                references_dict = apply_substitutions(references_dict, substitution_mapping)
 
                # Process each PDF in the references_dict
                found_pieces_summary = {}
                for pdf_data in references_dict:
                    for file_name, search_data in pdf_data.items():
                        print(file_name)
                        pdf_path = os.path.join("files", file_name)
                        found_pieces, pdf_bytes = incremental_search_and_highlight(pdf_path, search_data)
                        # print(pdf_bytes)
                        # found_pieces_summary[file_name]["found_pieces"] = found_pieces
                        # found_pieces_summary[file_name]["pdf_bytes"] = base64.b64encode(pdf_bytes).decode('utf-8')
                        found_pieces_summary[file_name] = base64.b64encode(pdf_bytes).decode('utf-8')
               
                # print(found_pieces_summary[0])
                return found_pieces_summary
 
    except Exception as e:
        print("Error in main:", str(e))
 
# # Run the main function
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)