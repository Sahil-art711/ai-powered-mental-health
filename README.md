# ai-powered-mental-health
# STEP 1: Install Required Libraries
!pip install -q transformers
!pip install -q torch

# STEP 2: Import Libraries
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pretrained DialoGPT model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For tracking conversation history
chat_history_ids = None

# Custom print with typing effect
def slow_print(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.02)
    print()

# Use DialoGPT for intelligent responses
def genai_response(user_input):
    global chat_history_ids
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Greeting
def greeting():
    slow_print("Hi, I’m MindMate 🤖—your AI-powered mental wellness buddy.")
    slow_print("How are you feeling today?")

# Mood detection
def mood_check():
    user_input = input("You: ").lower()
    if any(word in user_input for word in ["anxious", "stressed", "nervous"]):
        return "anxiety", user_input
    elif any(word in user_input for word in ["sad", "depressed", "down"]):
        return "sad", user_input
    elif any(word in user_input for word in ["happy", "good", "great"]):
        return "happy", user_input
    else:
        return "unknown", user_input

# Anxiety support
def anxiety_support():
    slow_print("I'm sorry you're feeling anxious. Let's try a grounding exercise.")
    slow_print("Name 5 things you can see around you:")
    input("> ")
    slow_print("Now 4 things you can touch:")
    input("> ")
    slow_print("3 things you can hear:")
    input("> ")
    slow_print("2 things you can smell:")
    input("> ")
    slow_print("1 thing you can taste:")
    input("> ")
    slow_print("Nice work grounding yourself 🌿")

# Sad support
def sad_support():
    slow_print("I'm here for you. Do you want to talk about it or try a calming activity?")
    choice = input("Type 'talk' or 'activity': ").lower()
    if choice == "talk":
        slow_print("Go ahead, I'm listening...")
        user_input = input("> ")
        response = genai_response(user_input)
        slow_print("Bot: " + response)
    else:
        slow_print("Let's try a breathing exercise together.")
        slow_print("Breathe in for 4 seconds...")
        time.sleep(4)
        slow_print("Hold it for 4 seconds...")
        time.sleep(4)
        slow_print("Now exhale slowly for 6 seconds...")
        time.sleep(6)
        slow_print("Feel a little calmer? 😊")

# Journal entry
def journal_prompt():
    slow_print("Would you like to write a quick journal entry? (yes/no)")
    if input("> ").lower() == "yes":
        slow_print("What's one thing you’re grateful for today?")
        entry = input("> ")
        with open("journal.txt", "a") as file:
            file.write(entry + "\n")
        slow_print("Entry saved. Gratitude makes a difference ✨")

# Crisis help
def crisis_check():
    slow_print("⚠️ If you're in crisis, please talk to a professional or call a helpline.")
    slow_print("📞 National Suicide Prevention Lifeline (US): 1-800-273-TALK (8255)")
    slow_print("🌐 You can also search online for help in your country.")

# Main function
def main():
    greeting()
    mood, first_input = mood_check()

    if mood == "anxiety":
        anxiety_support()
    elif mood == "sad":
        sad_support()
    elif mood == "happy":
        slow_print("That's great to hear! Keep spreading that positive energy 💛")
    else:
        slow_print("Thanks for sharing. Let’s talk about it...")
        response = genai_response(first_input)
        slow_print("Bot: " + response)

    journal_prompt()
    crisis_check()
    slow_print("Take care. You matter 💙 See you next time!")

# Run
if __name__ == "__main__":
    main()
pip install gtts playsound
from gtts import gTTS
import IPython.display as ipd

def speak_response(text):
    # Create speech
    tts = gTTS(text=text, lang='en', slow=False)
    # Save to a file
    tts.save("response.mp3")
    # Play audio
    return ipd.Audio("response.mp3", autoplay=True)

# Example chatbot response
ai_response = "hello! i am your ai chatbot. how can i help you? don't panic! Take deep breathe. I'm here with you."

# Call function
speak_response(ai_response)
!pip install gTTS transformers torch sentencepiece gradio
!pip install openai
!pip install SpeechRecognition pydub
import gradio as gr
from gtts import gTTS
from transformers import pipeline
import openai
import os
import random
from IPython.display import Audio
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your API key
emotion_classifier = pipeline('text-classification', model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
# Detect emotion
def detect_emotion(user_input):
    result = emotion_classifier(user_input)
    emotion = result[0]['label']
    return emotion

# Generate chatbot response
def generate_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can change to "gpt-4" if you want
        messages=[{"role": "system", "content": "You are an empathetic mental health support bot."},
                  {"role": "user", "content": user_input}]
    )
    reply = response['choices'][0]['message']['content']
    return reply

# Crisis detection
def detect_crisis(text):
    crisis_keywords = ["suicide", "kill myself", "can't live", "end my life", "hurt myself"]
    return any(word in text.lower() for word in crisis_keywords)

# Text-to-Speech (gTTS)
def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    return filename
def chat_with_bot(user_input):
    # Crisis check
    if detect_crisis(user_input):
        urgent_message = "I'm detecting that you might be in distress. Please contact a mental health professional or helpline immediately."
        speech_file = text_to_speech(urgent_message)
        return urgent_message, speech_file

    # Detect emotion
    emotion = detect_emotion(user_input)

    # Generate bot reply
    bot_response = generate_response(user_input)

    # Adjust voice speed depending on emotion (optional)
    speech_file = text_to_speech(bot_response)

    return f"Emotion detected: {emotion}\n\nBot: {bot_response}", speech_file
interface = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(lines=2, placeholder="How are you feeling today?"),
    outputs=[gr.Textbox(), gr.Audio()],
    title="AI Mental Health Support System",
    description="Talk to an AI counselor. The AI will detect emotions, generate a response, and speak to you."
)

interface.launch()
pip install openai-whisper
import whisper

model = whisper.load_model("base")  # or "small", "medium", "large"
def transcribe_audio(path):
    result = model.transcribe(path)
    return result["text"]
pip install openai-whisper gtts openai sounddevice scipy
!pip install openai-whisper gtts openai pydub ffmpeg-python
from IPython.display import Audio, display
import IPython
import io
import os
import openai
import whisper
from gtts import gTTS
from pydub import AudioSegment

# Set your OpenAI key
openai.api_key = "your_openai_api_key"

# Recording tool for Colab (run this cell to record)
from google.colab import output
from base64 import b64decode
from IPython.display import HTML

RECORD = """
<script>
var my_recorder = new Object();
my_recorder.submit = function(){
    var recorder = document.querySelector('#audio-recorder');
    var data = recorder.getAttribute('data-audio');
    google.colab.kernel.invokeFunction('notebook.upload_audio', [data], {});
}
</script>
<details>
<summary>🎙️ Click to Record Your Voice</summary>
<record-audio id="audio-recorder" controls></record-audio><br>
<button onclick="my_recorder.submit()">Upload</button>
</details>
"""

def record_audio():
    display(HTML(RECORD))

audio_path = "user_input.wav"

def decode_audio(data):
    audio = b64decode(data.split(',')[1])
    with open(audio_path, 'wb') as f:
        f.write(audio)

output.register_callback('notebook.upload_audio', decode_audio)
record_audio()
from google.colab import output
from IPython.display import Javascript, display
from base64 import b64decode
import io

RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time))

var record = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks = [];

  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = async () => {
    const blob = new Blob(chunks);
    const arrayBuffer = await blob.arrayBuffer();
    const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
    google.colab.kernel.invokeFunction('notebook.upload_audio', [base64], {});
  };

  recorder.start();
  await sleep(5000); // 5 seconds
  recorder.stop();
};

record();
"""

def decode_audio(x):
    audio_bytes = b64decode(x)
    with open("user_input.wav", "wb") as f:
        f.write(audio_bytes)

output.register_callback("notebook.upload_audio", decode_audio)
display(Javascript(RECORD_JS))
import whisper

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the recorded audio
result = model.transcribe("user_input.wav")
print("You said:", result["text"])
def get_gpt_response(user_input):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or a suitable replacement engine
        prompt=user_input,
        max_tokens=150  # Adjust as needed
    )
    reply = response.choices[0].text.strip()
    print("🤖 GPT says:", reply)
    return replyfrom gtts import gTTS
from IPython.display import Audio

# 1. Get voice-transcribed user input (assume it's stored in `user_input`)
# For example, from Whisper:
# user_input = result["text"]
user_input = result["I am worried about my ants and htir exams."]

# 2. Get GPT's response
gpt_reply = get_gpt_response(user_input)

# 3. Convert GPT reply to speech using gTTS
tts = gTTS(text=gpt_reply, lang='en')
tts.save("gpt_reply.mp3")

# 4. Play the spoken reply
Audio("gpt_reply.mp3", autoplay=True)
pip install --upgrade openai# ENOJIS MOOD DETECTOR# Ask user for their mood
mood = input("How are you feeling right now? 😊😔😡😨😐\n(Use emoji or words): ")

# Simple logic to respond based on mood
if "😊" in mood or "happy" in mood:
    print("That's great! Keep up the positive vibes! 🌟")
elif "😔" in mood or "sad" in mood:
    print("I'm here for you. Want to talk about it?")
elif "😡" in mood or "angry" in mood:
    print("Try deep breathing. Want to do a quick exercise together?")
elif "😨" in mood or "anxious" in mood:
    print("You're not alone. Let’s try a grounding technique.")
else:
    print("Thank you for sharing. I'm here to listen. ❤️")pip install flask transformers torch nltk scikit-learn
# sentiment_analysis.py
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']
# cbt_response.py
def generate_cbt_response(label, text):
    if label == "POSITIVE":
        return "I'm glad you're feeling okay. Want to talk more about it?"
    elif label == "NEGATIVE":
        return (
            "It sounds like you're going through something tough. "
            "Would you like to try a breathing exercise or journaling prompt?"
        )
    else:
        return "I'm here for you. Want to talk more about what's on your mind?"
# tools.py
import random

def get_breathing_exercise():
    return "Let's try a deep breathing exercise. Breathe in for 4 seconds... hold... and exhale slowly."

def get_journaling_prompt():
    prompts = [
        "Write about a moment today you felt proud.",
        "What are 3 things you're grateful for right now?",
        "Describe how you feel using 3 words and explain why."
    ]
    return random.choice(prompts)

def get_affirmation():
    affirmations = [
        "You are doing your best, and that is enough.",
        "Your feelings are valid.",
        "This too shall pass."
    ]
    return random.choice(affirmations)
# crisis_detector.py
def detect_crisis(text):
    crisis_keywords = ["suicide", "kill myself", "end it all", "can't go on"]
    return any(keyword in text.lower() for keyword in crisis_keywords)

def emergency_response():
    return (
        "It sounds like you're in a lot of pain. Please know you're not alone. "
        "Here are some emergency contacts:\n\n"
        "📞 Suicide Prevention Hotline: 1-800-273-8255\n"
        "🆘 Call emergency services in your area immediately."
    )
import time

def breathing_exercise():
    print("🧘 Let’s do a 4-7-8 breathing exercise.")
    for i in range(3):  # repeat 3 times
        print("\nBreathe in... (4 sec)")
        time.sleep(4)
        print("Hold... (7 sec)")
        time.sleep(7)
        print("Breathe out... (8 sec)")
        time.sleep(8)
    print("\nFeel better? You’re doing great. 💙")

breathing_exercise()
import random

affirmations = [
    "You are enough just as you are.",
    "This moment is temporary; you are growing stronger.",
    "Your feelings are valid.",
    "You’ve made it through 100% of your worst days.",
    "You are capable of amazing things."
]

def give_affirmation():
    print("💬 Affirmation of the day:")
    print(random.choice(affirmations))

give_affirmation()def journal_entry():
    print("📝 Write how you're feeling today. (Press Enter when done)")
    entry = input("Your journal: ")
    with open("journal.txt", "a") as file:
        file.write("\n---\n")
        file.write(entry)
    print("Saved! It's healthy to express your feelings. 📔")

journal_entry()def is_session_end(text):
    """
    Check if user wants to end the conversation.
    """
    text = text.lower()
    closing_keywords = ["bye", "thank you", "thanks", "that's all", "goodbye", "i'm good now", "talk later"]
    return any(keyword in text for keyword in closing_keywords)

def closing_message():
    """
    Returns a comforting closure message.
    """
    return (
        "💬 I'm really glad you reached out today. Remember, your feelings are valid.\n"
        "🧘‍♀️ Take some time for yourself, and check in with your emotions when you can.\n"
        "🌟 You are stronger than you think. I'm always here when you need to talk.\n\n"
        "👋 Until next time!"
    )
import re

def is_session_end(text):
    """
    Detects if the user's message indicates an intent to end the session.
    """
    text = text.lower()

    # Patterns to detect various ways of saying goodbye or thanks
    patterns = [
        r'\bbye\b',
        r'\bgoodbye\b',
        r'\bthank(s| you)?\b',
        r"that's all",
        r"talk to you later",
        r"i'm good now",
        r"see you",
        r"catch you later",
        r"talk later"
    ]

    return any(re.search(pattern, text) for pattern in patterns)
