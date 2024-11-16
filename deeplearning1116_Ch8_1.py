import speech_recognition as sr

# Create a recognizer object
recognizer = sr.Recognizer()

# Use the microphone as the input device
with sr.Microphone() as source:
    print("Please start speaking...")
    # Adjust the recognizer to the ambient noise level
    recognizer.adjust_for_ambient_noise(source)

    # Capture the user's speech input
    audio = recognizer.listen(source)

    try:
        # Convert speech to text using Google Speech Recognition API
        text = recognizer.recognize_google(audio, language='zh-TW') # chinese
        # text = recognizer.recognize_google(audio, language='en-US')
        print("Recognized text:", text)
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not connect to Google Speech API: {e}")
