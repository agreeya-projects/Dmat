import os
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create a speech recognition object
r = sr.Recognizer()

def transcribe_audio(path):
    """
    Recognizes speech in the audio file and converts it to text.
    
    Parameters:
    path (str): Path to the input audio file.
    
    Returns:
    str: Transcribed text from the audio file.
    """
    with sr.AudioFile(path) as source:
        audio_listened = r.record(source)
        text = ""
        try:
            text = r.recognize_google(audio_listened)
        except sr.UnknownValueError as e:
            print(f"Error recognizing {path}: {e}")
        return text

def get_large_audio_transcription_on_silence(path, chunks_folder):
    """
    Splits the audio file into chunks on silence and applies speech recognition on each chunk.
    
    Parameters:
    path (str): Path to the input audio file.
    chunks_folder (str): Path to the folder where audio chunks will be saved.
    
    Returns:
    str: Transcribed text from all audio chunks.
    """
    sound = AudioSegment.from_file(path)
    chunks = split_on_silence(sound,
                              min_silence_len=500,
                              silence_thresh=sound.dBFS-14,
                              keep_silence=500)

    if not os.path.isdir(chunks_folder):
        os.makedirs(chunks_folder)

    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(chunks_folder, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        text = transcribe_audio(chunk_filename)
        text = f"{text.capitalize()}. " if text else ""
        print(chunk_filename, ":", text)
        whole_text += text

    return whole_text

def process_audio_files_in_folder(base_folder, text_folder, max_workers=8):
    """
    Processes all audio files in a folder, converting them to text files in a specified folder.
    
    Parameters:
    base_folder (str): Path to the base folder containing subfolders with audio files.
    text_folder (str): Path to the folder where text files will be saved.
    max_workers (int): Maximum number of concurrent threads (default is 8).
    """
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)
        print(f"Created text folder: {text_folder}")

    for root, dirs, files in os.walk(base_folder):
        audio_files = [f for f in files if f.endswith(".wav")]
        if not audio_files:
            continue

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_single_audio_file, 
                    os.path.join(root, filename), 
                    text_folder, 
                    os.path.join(root, os.path.splitext(filename)[0] + "_chunks"),
                    os.path.relpath(root, base_folder)
                ): filename
                for filename in audio_files
            }

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred for {filename}: {e}")

def process_single_audio_file(audio_file, text_folder, chunks_folder, relative_path):
    """
    Processes a single audio file, converting it to a text file in a specified folder.
    
    Parameters:
    audio_file (str): Path to the input audio file.
    text_folder (str): Path to the folder where the text file will be saved.
    chunks_folder (str): Path to the folder where audio chunks will be saved.
    relative_path (str): Relative path from base folder to the subfolder containing the audio file.
    """
    full_text = get_large_audio_transcription_on_silence(audio_file, chunks_folder)
    relative_folder = relative_path.replace(os.sep, "_")
    text_file_name = os.path.join(text_folder, f"{relative_folder}_{os.path.splitext(os.path.basename(audio_file))[0]}.txt")
    
    with open(text_file_name, "w") as text_file:
        text_file.write(full_text)
        print(f"Saved transcription to {text_file_name}")
    
    # Delete the chunks folder after processing
    if os.path.exists(chunks_folder):
        shutil.rmtree(chunks_folder)
        print(f"Deleted chunks folder: {chunks_folder}")

if __name__ == "__main__":
    base_folder = "all_data"
    text_folder = "all_data/text"
    print(f"Starting transcription of audio files in {base_folder} to {text_folder}")
    process_audio_files_in_folder(base_folder, text_folder)
    print("All transcriptions are complete.")
