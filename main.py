from moviepy.editor import VideoFileClip
import os
import whisper

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from video file and save as WAV"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path, language="en"):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("turbo")  # Options: "tiny", "base", "small", "medium", or "large" (I recommend "turbo")
        
        print(f"Transcribing audio in {language}...")
        result = model.transcribe(audio_path, language=language)
        return result["text"]
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def video_to_text(video_path, output_file="transcription.txt", language="en"):
    """Main function to convert video to text"""
    try:
        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = extract_audio(video_path)
        
        # Transcribe audio
        print("\nStarting transcription...")
        transcription = transcribe_audio(audio_path, language)
        
        if not transcription:
            print("Transcription failed!")
            return
        
        # Clean up 
        try:
            os.remove(audio_path)
        except OSError:
            print(f"Warning: Could not remove temporary file {audio_path}")
        
        # Save transcription 
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print(f"\nTranscription complete! Saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    video_path = (input("Enter video file path: ")).strip()
    language = input("Enter language code (e.g., 'en' for English, 'pt' for Portuguese, default: en): ").strip() or "en"
    output_file = ("results/"+input("Enter output file name (default: transcription.txt): ")+".txt").strip() or "results/transcription.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
    else:
        video_to_text(video_path, output_file, language)