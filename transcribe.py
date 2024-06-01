def extract_audio(input_path):
    import moviepy.editor as mp
    import os
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        audio_path = "extracted_audio.wav"
        video = mp.VideoFileClip(input_path)
        video.audio.write_audiofile(audio_path)
        return audio_path
    elif file_extension in ['.wav', '.mp3', '.flac', '.aac']:
        return input_path
    else:
        raise ValueError("Unsupported file format")

