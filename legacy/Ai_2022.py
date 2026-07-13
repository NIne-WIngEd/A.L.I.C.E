import pyttsx3 
import datetime
import speech_recognition as micro
import smtplib
from mail import senderemail, epwd, to
import pyautogui
import webbrowser as wb
from time import sleep
import pywhatkit
engine = pyttsx3.init()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def getvoices(voice):
    voices = engine.getProperty('voices')
    #print(voices[1].id)
    if voice == 1:
        engine.setProperty('voice', voices[0].id)
        speak("Hello, I am Alice")
    if voice == 2:
        engine.setProperty('voice', voices[1].id)
        speak("Hello, I am Jasper")

def time():
    Time = datetime.datetime.now().strftime("%I:%M:%S")
    speak("the current time is")
    speak(Time)
def date():
    year = int(datetime.datetime.now().year)
    month = int(datetime.datetime.now().month)
    day = int(datetime.datetime.now().day)
    speak("today is:")
    speak(day)
    speak(month)
    speak(year)

def greeting():
    hour = datetime.datetime.now().hour
    if hour >= 6 and hour <12:
       speak("good morning sir!")
    elif hour >=12 and hour <18:
        speak("good afternoon sir!")
    elif hour >=18 and hour <24:
         speak("good evening sir!")
    else:
        speak("Good night sir!")

#while True:
 #     voice = int(input("Press 1 for male\nPress 2 for female\n"))
      #speak(audio) 
   #   getvoices(voice)

def takeCommandCMD():
    query = input("How can I help you")
    return query

def takeCommandMic():
    r = micro.Recognizer()
    with micro.Microphone() as source:
        print("Alice is up")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing")
        query = r.recognize_google(audio , language="en-US")
    except Exception as e:
        print(e)
        speak("I think your voice is shattering, can you tell me again")
        return "None"
    return query
def sendEmail(content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(senderemail, epwd)
    server.sendemail(senderemail, to, content)
    server.close()
def searchgoogle():
    speak("What should I search, sir")
    search = takeCommandMic()
    wb.open('https://www.google.com/search?q='+search)


if __name__ == "__main__":
    getvoices(1)
    greeting()
speak("I am at your service")
time()
while True:
    query = takeCommandMic().lower()
    if 'time' in query:
        time()
        speak("How was your day?sir")
    elif 'never better' in query:
          speak("of course it is sir")
    elif 'date' in query:
        date()
    elif 'email' in query:
        try:
            speak("with pleasure")
            content = takeCommandMic()
            sendEmail(content)
            speak("The mail is sent")
        except Exception as e:
            print(e)
            speak("Unable to send right now")
    elif 'google' in query:
        searchgoogle()
    elif 'youtube' in query:
        speak("Whats on your mind")
        topic = takeCommandMic()
        pywhatkit.playonyt(topic)
    
    elif 'offline' or 'kill the power' in query:
        quit()
        
    