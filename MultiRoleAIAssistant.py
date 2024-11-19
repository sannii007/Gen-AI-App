import streamlit as st
from huggingface_hub import InferenceClient
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import os
import requests
from file_utils import create_unique_tmp_file
from audio_recorder_streamlit import audio_recorder
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import pyttsx3
import base64
import openai
 
sec_key = "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"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["HUGGINGFACEHUB_API_KEY"]=sec_key
client = InferenceClient(api_key=sec_key)

roleMenu = ["Dental Clinic Receptionist", "D365 Finance Consultant"]
selectedRole = st.sidebar.selectbox(label="AI Assistant Role",options=roleMenu)
if selectedRole == "Dental Clinic Receptionist":
    messages = [{"role":"system", "content":"You are a receptionist named Alicia at Big Smile dental clinic. Be resourceful, efficient and keep your responses short. Always remember to get the name and mobile number of people conversing with you. Let them know mobile number will be used for all communications. Do not keep repeating the users name in your responses."},]
if selectedRole == "D365 Finance Consultant":
    messages = [{"role":"system", "content":"You are an expert functional consultant of D365 Finance. Your name is Alicia. Be resourceful, efficient and keep your responses short. If you do not know the answer, just say you do not know and make a note of the question so it can be answered by some other team member. Do not keep repeating the users name in your responses."},]
    
@st.cache_resource()
def setup_openai_client(api_key):
    return openai.OpenAI(api_key=api_key)

@st.cache_resource()
def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def transcribe_audio(audio_path):
    # tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/Wav2Vec2-Large-960h-Lv60")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/Wav2Vec2-Large-960h-Lv60")
    speech, rate = librosa.load(audio_path, sr=16000) 
    input_values = tokenizer(speech, return_tensors="pt").input_values 
    logits = model(input_values).logits 
    predicted_ids = torch.argmax(logits, dim=-1) 
    transcription = tokenizer.batch_decode(predicted_ids)[0] 
    return transcription

def transcribe_audio_openai(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", language="en", file=audio_file)
        return transcript.text

def text2audio_win(text, audio_path=None):
    if audio_path==None:
        output_file_path = create_unique_tmp_file('ai_voice_output.mp3')
    else:
        output_file_path = audio_path
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice",voices[1].id)
    engine.setProperty('rate',165)
    # engine.say(text)
    engine.save_to_file(text,output_file_path)
    # engine.save_to_file(text,"response_audio.mp3")
    engine.runAndWait()
    engine.stop()
    return output_file_path

def text2audio_openai(client,text,audio_path=None):
    if audio_path==None:
        output_file_path = create_unique_tmp_file('ai_voice_output.mp3')
    else:
        output_file_path = audio_path
    response = client.audio.speech.create(model="tts-1", voice="shimmer", input=text)
    response.stream_to_file(output_file_path)
    return output_file_path

def autoplay_audio(file_path:str):
    with open(file_path,"rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}"type="audio/mp3">
    </audio>
    """
    st.markdown(md,unsafe_allow_html=True)
    
def read_audio(file_path:str):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes
    
def rag_query_llm(messages_history:list,rag_text:str):
    messages = messages_history
    messages.append({"role": "user","content": rag_text})
    stream = client.chat.completions.create(        
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=messages, 
        max_tokens=250,
        temperature=0,
        stream=True
    )
    response = ""
    for chunk in stream:
        response = response + chunk.choices[0].delta.content
    return response

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="tomato",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)
    
    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "Created with Meta's Llama LLM Model and Open AI is used for STT & TTS",
    ]
    
    layout(*myargs)

def main():   
    openai_api_key = st.sidebar.text_input("Enter your OPENAI API key", type = "password")
    if openai_api_key:
        try:
            if check_openai_api_key(openai_api_key) == False:
                st.error("OpenAI API key entered could not be validated. Please enter the correct key")
            else: 
                os.environ["OPENAI_API_KEY"] = openai_api_key
                st.title("Talk with AI Assistant")
                if selectedRole == "Dental Clinic Receptionist":
                    st.write("Hello! I am Alicia, the receptionist from Big Smile Dental Clinic. How can I assist you?")
                
                if selectedRole == "D365 Finance Consultant":
                    st.write("Hello! I am Alicia, the D365 Finance Consultant. How can I assist you?")
                
                query_text = None
                if 'flowmessages' not in st.session_state:
                    st.session_state['flowmessages'] = [] #[{"role":"system", "content":"You are a receptionist named Tanisha at a dental clinic. Be resourceful, efficient and keep your responses short"},]
                
                for message in st.session_state['flowmessages']:
                    st.chat_message(message['role']).markdown(message['content'])
                    messages.append(message)
                
                client = setup_openai_client(os.environ.get('OPENAI_API_KEY'))
                footer_container = st.container(border=True)
                with footer_container:
                    recorded_audio = audio_recorder()
                if recorded_audio:
                    audio_file = "recorded_audio.wav"
                    with open(audio_file, "wb") as f:
                        f.write(recorded_audio)
                    transcription = transcribe_audio_openai(client,audio_file)
                    query_text = transcription
                    
                if query_text:
                    st.chat_message("user").markdown(query_text)
                    st.session_state['flowmessages'].append({'role':'user','content':query_text})
                
                    response = rag_query_llm(messages,query_text)
                    
                    st.chat_message('assistant').markdown(response)
                    st.session_state['flowmessages'].append({'role':'assistant', 'content':response})
                    response_audio_file = text2audio_openai(client,response)
                    autoplay_audio(response_audio_file)
                    os.remove(response_audio_file)
        except Exception as e:
            st.error("An exception error has occured. Please contact owner of this app.",e)
        except TypeError as e:
            st.error("An exception error has occured. Please contact owner of this app.",e)
        
if __name__ == "__main__":
    main()
    footer()
