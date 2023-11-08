from flask import Flask, request, render_template, send_from_directory, redirect, url_for,jsonify
from pathlib import Path
from pytube import YouTube
import os
import torch
import whisper
from whisper.utils import get_writer
from werkzeug.utils import secure_filename
from flask import send_file
from zipfile import ZipFile
from claude_api import Client
import time
from datetime import datetime
import random
import json
import markdown
import re
import config

app = Flask(__name__)
# Initialize your Claude API client with the session cookie

cookie = config.cookie
# Use CUDA, if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# Load the Whisper model and other functions here
model = whisper.load_model("base").to(DEVICE)
print("Load model Successfully")

# Define the path for uploading and storing files
UPLOAD_FOLDER = './temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Max file size (50MB)
ALLOWED_EXTENSIONS = set(['pdf','txt', 'png', 'jpg', 'jpeg', 'gif','mp3','mp4','m4a'])
database = {
    'temp.txt': './temp/temp.txt',
    'temp.md': './temp/temp.md',
    'temp.srt': './temp/temp.srt',
    'temp.vtt': './temp/temp.vtt',
    'temp.tsv': './temp/temp.tsv',
}
def to_snake_case(name):
    return name.lower().replace(" ", "_").replace(":", "_").replace("__", "_")

def download_youtube_audio(url,  file_name = None):
    "Download the audio from a YouTube video"
    target_path = "./temp/"

    yt = YouTube(url)

    video = yt.streams.filter(only_audio=True).first()

    out_file = video.download(output_path=target_path)

    video_title, ext = os.path.splitext(out_file)
    file_name = video_title + '.mp3'
    os.rename(out_file, file_name)

    print("target path = " + (file_name))
    print("mp3 has been successfully downloaded.")
    return file_name
def download_multiple_files(files_to_send):
    zip_filename = './temp/transripts.zip'  # Name for the zip file
    with ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_send:
            zipf.write(file)  # Add each file to the zip
            os.remove(file)
    response = send_file(zip_filename, as_attachment=True)
    os.remove(zip_filename)  # Remove the temporary zip file

    return response

def save_response_to_markdown(filename,response,files_to_send):
    # Create a unique filename (e.g., using a timestamp)
    name = filename.split('.')[0]
    response_filename = f"{name}.md"
    # Specify the directory to save the file (e.g., within the UPLOAD_FOLDER)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], response_filename)
    
    # Save the response as a Markdown file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(response)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.md'), 'w', encoding='utf-8') as file:
        file.write(response)
    files_to_send.append(file_path)
    
    
    return response_filename 

def claude(file,filename,files_to_send):
    claude_api = Client(cookie)
    print("Sending....",filename)
    # you can create your own prompt here
    prompt = "這份txt檔是一段我自己整理的語音轉文字的逐字稿。我需要詳細內容摘要並以Markdown格式輸出。以繁體中文回答"
    conversation_id = claude_api.create_new_chat()['uuid']
    
    response = claude_api.send_message(prompt, conversation_id,attachment=file,timeout=600)
    response_filename = save_response_to_markdown(filename,response,files_to_send)

    deleted = claude_api.delete_conversation(conversation_id)
    if deleted:
        print("Conversation deleted successfully")
    else:
        print("Failed to delete conversation")
@app.route('/transcribe_file', methods=['POST'])
def transcribe_file(model, file, plain, srt, vtt, tsv,summarize):
    """
    Runs Whisper on an audio file
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    files_to_send = []
    file_path = Path(file)
    output_directory = file_path.parent
    
    # Run Whisper
    result = model.transcribe(file, verbose = False)

    # Set some initial options values
    options = {
        'max_line_width': None,
        'max_line_count': None,
        'highlight_words': False
    }
    print(f"\nCreating text file")

    #Record the upload status
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open('./temp/upload-record.txt','a') as f:
        f.write(current_time+': '+str(file_path.stem)+'\n')

    # Save as a TXT file with hard line breaks
    txt_writer = get_writer("txt", output_directory)
    txt_writer(result, str(file_path.stem),options)
    txt_writer(result, 'temp',options)
    
    
    if plain:
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.txt')
        txt_writer(result, 'temp',options)
        files_to_send.append(result_file)
    if srt:
        print(f"\nCreating SRT file")
        srt_writer = get_writer("srt", output_directory)
        srt_writer(result, str(file_path.stem), options)
        srt_writer(result, 'temp',options)
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.srt')
        files_to_send.append(result_file)
    if vtt:
        print(f"\nCreating VTT file")
        vtt_writer = get_writer("vtt", output_directory)
        vtt_writer(result, str(file_path.stem), options)
        vtt_writer(result, 'temp', options)
        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.vtt')
        files_to_send.append(result_file)
    if tsv:
        print(f"\nCreating TSV file")

        tsv_writer = get_writer("tsv", output_directory)
        tsv_writer(result, str(file_path.stem), options)
        tsv_writer(result, 'temp', options)

        result_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.tsv')
        files_to_send.append(result_file)
    if summarize:
        txt_file = os.path.join(app.config['UPLOAD_FOLDER'], str(file_path.stem)+'.txt')
        try:
            claude(txt_file,str(file_path.stem),files_to_send)
        except:
            print("sorry this api reach limits")

    # Clean up the temporary file
    os.remove(file)
    response = download_multiple_files(files_to_send)
    return response

@app.route('/')
def index():
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@app.route('/transcribe', methods=['POST','GET'])
def transcribe():
    if request.method == 'POST':
        

        input_format = request.form.get('input_format')
        plain = request.form.get('plain') == 'true'
        srt = request.form.get('srt') == 'true'
        vtt = request.form.get('vtt') == 'true'
        tsv = request.form.get('tsv') == 'true'
        summarize = request.form.get('summarize') == 'true'
        
        result = None  # Define a default value for result
        file = None  # Define a default value for file
        
        if input_format == 'youtube':
            # Handle YouTube transcription
            url = request.form.get('url')
            # Implement the YouTube transcription logic here
            # Download the audio stream of the YouTube video
            audio = download_youtube_audio(url)
            print(f"Downloading audio stream: {audio}")
            time.sleep(1)
            # Transcribe the audio stream
            result = transcribe_file(model, audio, plain, srt, vtt, tsv,summarize)
        else:
            if input_format == 'mp3':
            # Handle local file transcription
                file = request.files['mp3_file']
            elif input_format == 'm4a':
                file = request.files['m4a_file']
            print("success upload")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                result = transcribe_file(model, os.path.join(app.config['UPLOAD_FOLDER'], filename), plain, srt, vtt, tsv,summarize)   
        
        # Sample list of files for demonstration
        file_list = []
        if(summarize):
            file_list.append('summary')
        if(plain):
            file_list.append('txt')
        if(srt):
            file_list.append('srt')    
        if(vtt):
            file_list.append('vtt')
        if(tsv):
            file_list.append('tsv')
        # 将列表转换为JSON字符串
        file_list_str = json.dumps(file_list)     
        return redirect(url_for('select_result',file_list=file_list_str))
    return redirect(url_for('index'))


@app.route('/select_result', methods=['GET', 'POST'])
def select_result():
    file_list_str = request.args.get('file_list', default="[]")
    # 解析JSON字符串为Python列表
    file_list = json.loads(file_list_str)
    if request.method == 'POST':
        selected_file = request.form['selected_file']
        # You can now use the selected_file to perform actions on the chosen file
        if selected_file=='summary':
            selected_file='md'
        return redirect(url_for('view_file',filename="temp."+selected_file,file_list=file_list_str))

    return render_template('select_file.html', selectfile_list=file_list)

@app.route('/result/<filename>')
def view_file(filename):
    file_list_str = request.args.get('file_list', default="[]")
    # 解析JSON字符串为Python列表
    file_list = json.loads(file_list_str)
    import urllib.parse
    url_encoded = urllib.parse.quote(str(file_list).replace("'", '"'))
    # 解析JSON字符串为Python列表
    if filename in database:
        file_path = database[filename]
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                if filename.endswith('.md'):
                    
                    markdown_text = file.read()
                    content = markdown.markdown(markdown_text, extensions=['extra', 'toc', 'tables', 'codehilite', 'fenced_code'])
                else:
                    content = file.read()
            return render_template('file_content.html', content=content,url_encoded=url_encoded)
        else:
            return "文件不存在。"
    else:
        return "非法文件请求。"
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)