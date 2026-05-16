import os, csv
import av
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# CONFIGURATION
BASE_DIRECTORY = "."
INPUT_AUDIO = "sample_audio_clips/sample_audio.mp3"


# Helper function: audio loader
def load_audio_with_av(filepath):
    container = av.open(filepath)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    
    audio_frames = []
    
    for frame in container.decode(audio=0):
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            audio_frames.append(rf.to_ndarray())
    
    waveform = np.concatenate(audio_frames, axis=1).reshape(-1)
    waveform = waveform.astype(np.float32)

    return waveform


# Helper function: model loader
def load_yamnet():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    class_map_path = model.class_map_path().numpy()
    class_names = []
    
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    
    return model, class_names


# Load audio clip and model, then run inference and return model outputs
def run_inference(input_filepath):
    waveform = load_audio_with_av(input_filepath)
    model, class_names = load_yamnet()

    # YAMNet forward pass for inference; returns scores, embeddings, and spectrogram
    scores, embeddings, spectrogram = model(waveform)

    scores = scores.numpy()
    embeddings = embeddings.numpy()
    spectrogram = spectrogram.numpy()

    return {
        "scores": scores,
        "embeddings": embeddings,
        "spectrogram": spectrogram,
        "class_names": class_names
    }


# Helper function: print frame-level predictions to terminal
def print_frame_predictions(outputs, top_k=3, every_n_frames=1):
    # YAMNet's default internal hop interval
    default_yamnet_frame_hop_seconds = 0.48
    
    scores = outputs["scores"]
    class_names = outputs["class_names"]

    for frame_idx in range(0, scores.shape[0], every_n_frames):
        time_sec = frame_idx * default_yamnet_frame_hop_seconds
        top_indices = np.argsort(scores[frame_idx])[::-1][:top_k]
        
        print(f"\nFrame {frame_idx} at ~{time_sec:.2f}s")
        for idx in top_indices:
            print(f"\t{class_names[idx]}: {scores[frame_idx, idx]:.3f}")


# Main
if __name__ == "__main__":
    input_path = os.path.join(BASE_DIRECTORY, INPUT_AUDIO)
    
    # You can extract and save the embeddings and prediction class scores from outputs 
    outputs = run_inference(input_path)

    print("YAMNet outputs:")
    print("scores shape:", outputs["scores"].shape)
    print("embeddings shape:", outputs["embeddings"].shape)
    print("spectrogram shape:", outputs["spectrogram"].shape)

    print_frame_predictions(outputs, top_k=3)
