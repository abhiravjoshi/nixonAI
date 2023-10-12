from elevenlabs import clone, generate, play, set_api_key, save
import os
import random
from api_handler import delete_voices, print_voices

# source for using OpenAI with ElevenLabs: https://betterprogramming.pub/talking-with-ai-using-whisper-gpt-3-and-speech-synthesis-2b86ce7c5c59

# Processing training data and creating txt file corpus
with open("apikey3.txt", 'r') as f:
    API_KEY = f.read()

set_api_key(API_KEY)

nixon_files = ['dataset/training/' + f for f in os.listdir('dataset/training')]
if 'dataset/training/.DS_Store' in nixon_files:
    nixon_files.remove('dataset/training/.DS_Store')
nixon_files = sorted(nixon_files)
# print(nixon_files)
# exit()

print_voices()


if False: # Make this false once you're happy with the Nixon clone
    
    # delete_voices("Nixon3") # will delete any previous versions of Nixon voice in API.
    
    # .3 .3 .3 Nixon
    
    voice = clone(
        name="Nixon3",
        stability=.3,
        similarity_boost=.3,
        style=.3,
        api_key=API_KEY,
        description="It's ya boy Tricky Dick", # Optional
        files=nixon_files,
        use_speaker_boost=True
    )

audio = generate(
    # text="As the world - turns... We must continue... to move forward... with the loss of this Titan.",
    text="You... once a part of me, now we don't even speak. - Tell me baby... do you miss me?",
    voice="Nixon"
    )

play(audio)
save(audio, "sample/sample.wav")