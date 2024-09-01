import os
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed

def video_to_audio(video_file, audio_file):
    try:
        video = VideoFileClip(video_file)
        video.audio.write_audiofile(audio_file)
        print(f"Audio extraction complete for {video_file}")
    except Exception as e:
        print(f"Failed to process {video_file}: {e}")

def process_files(video_folder, audio_folder, max_workers=10):
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    
    video_files = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith((".mp4", ".mkv", ".avi")):
                video_files.append(os.path.join(root, file))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(video_to_audio, video_file, generate_audio_file_path(video_file, video_folder, audio_folder)): video_file
            for video_file in video_files
        }
        
        for future in as_completed(future_to_file):
            video_file = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred for {video_file}: {e}")

def generate_audio_file_path(video_file, video_folder, audio_folder):
    relative_path = os.path.relpath(video_file, video_folder)
    relative_dir = os.path.dirname(relative_path)
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    audio_file_name = f"{relative_dir.replace(os.sep, '_')}_{base_name}.wav"
    return os.path.join(audio_folder, audio_file_name)

#if __name__ == "__main__":
def VideoToaudio_converter():
    video_folder = "all_data"
    audio_folder = "all_data/audio"
    process_files(video_folder, audio_folder)
    print("All audio extractions are complete.")
