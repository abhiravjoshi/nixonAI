# nixon
Richard Nixon Voice AI
Goals:
1) With OpenAI, use Whisper to transcribe audio that someone says into a microphone. Then from whatever's transcribed, perform TTS, utilizing training data based off of Nixon's interview
    a) Initially make a working product with pre-recorded wav or mp3 files
        - Utilized 100 custom audio snippets from Nixon speeches
        - Also used pre-trained vits model as weight (cloned from repo: https://huggingface.co/coqui/XTTS-v1/tree/main)
    b) Then make a version that takes a recording
2) Next step: have any text output comes out of ChatGPT response to be spoken in Nixon voice (using CoquiTTS)
3) Final step: take in a bunch of text training from Nixon's books / speeches etc. then the AI model would have to respond like its Nixon

Collaborators:
    - Par./Abhirav Joshi (@abhiravjoshi)
    - Fly Montag/Sumedh Garimella (@hdemusg)