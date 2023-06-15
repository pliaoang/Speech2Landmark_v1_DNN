import cv2
import os
from moviepy.editor import VideoFileClip
import numpy as np
import librosa
import mediapipe as mp
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import pickle


def iterFolder(fun, **kwargs):
    src_path = kwargs["src_path"]
    dest_path = kwargs["dest_path"]
    print(src_path, dest_path)

    if "frame_idx" in kwargs:
        frame_idx = kwargs["frame_idx"]
        for idx, file in enumerate(os.listdir(src_path)):
            curr_s_path = f"{src_path}\{file}"
            curr_d_path = f"{dest_path}\{file}"
            fun(curr_s_path, curr_d_path, frame_idx[idx])
        return 0

    sample_idx = []
    for idx, file in enumerate(os.listdir(src_path)):
        curr_s_path = f"{src_path}\{file}"
        curr_d_path = f"{dest_path}\{file}"

        value = fun(curr_s_path, curr_d_path)
        if value is not None:
            sample_idx.append(value)

    return sample_idx


def video2audio(src_path, dest_path):
    video = VideoFileClip(src_path)
    audio = video.audio
    audio.write_audiofile(
        f"{dest_path.split('.')[0]}.wav", codec="pcm_s16le", ffmpeg_params=["-ac", "1"])


def mel_spec(src_path, dest_path):
    # hyperparameters
    # For signal processing
    sr = 16000  # Sampling rate.
    n_fft = 534  # fft points (samples)
    hop_length = 267  # This is dependent on the frame_shift.
    n_mels = 40  # Number of Mel banks to generate
    preemphasis = .97  # or None

    # Loading sound file
    y, orig_sr = librosa.load(src_path, sr=44100)
    # Change sr from 44.1k -> 16k
    # y.shape = (16000,0)
    y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # To log mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)

    # Transpose
    log_mel = log_mel.T.astype(np.float32)  # (T, 40)
    # print("Shape of mel: ", log_mel.shape)

    # deprecate last row for alignment
    log_mel = log_mel[:-1, :]

    # save data
    np.save(f"{dest_path.split('.')[0]}", log_mel)

    return log_mel.shape[0]  # number of sample


def landmark(*args):
    src_path, dest_path, frame_idx = args

    # 嘴唇座標點邊號
    lipsLMInfo = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185,
                  191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

    mp_face_mesh = mp.solutions.face_mesh
    # For static images:
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    # Path to the .mp4 file
    vidcap = cv2.VideoCapture(src_path)
    success, image = vidcap.read()  # get image (frame of video)
    img_idx = 0
    row = np.empty((0, 80))
    while success:
        image = cv2.flip(image, 1)
        # Get facemesh, Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:  # if face dectected
            for face_landmarks in results.multi_face_landmarks:
                col = np.empty(0, dtype=np.float32)
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx not in lipsLMInfo:
                        continue

                    x = landmark.x
                    y = landmark.y
                    # check if the value are we want to.
                    col = np.append(col, [x, y])

                row = np.vstack((row, col))

        # reading next frame from video
        success, image = vidcap.read()
        img_idx += 1

        if img_idx == frame_idx:
            break

    np.save(f"{dest_path.split('.')[0]}", row)



def merge_mel_with_norm(src_path):
    mel_all = np.empty((0, 40))
    for file in os.listdir(src_path):
        curr_path = f"{src_path}\\{file}"
        data = np.load(f"{curr_path}")
        mel_all = np.vstack((mel_all, data))

    print("Shape of mel after merge: ", mel_all.shape)

    # 標準化 mel spec
    # instantiate StandardScaler
    scaler = StandardScaler()
    # Apply z-scorce
    mel_all = scaler.fit_transform(mel_all)
    with open("scaler_mel.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save("norm_mel", mel_all)


def merge_landmark_move_center_with_norm(src_path):
    landmark_all = np.empty((0, 80))
    for file in os.listdir(src_path):
        curr_path = f"{src_path}\\{file}"
        data = np.load(f"{curr_path}")
        landmark_all = np.vstack((landmark_all, data))

    print("Shape of landmark after merge: ", landmark_all.shape)

    # mean of 78 and 308 is the x, y coordinate of center
    landmark_move = np.empty((0, 80))
    for line in landmark_all:
        # each loop for one image
        # get x, y coordinate
        x_mean = (line[16]+line[52]) / 2
        y_mean = (line[17]+line[53]) / 2

        line[::2] -= x_mean  # x coordinate
        line[1::2] -= y_mean  # y coordinate

        landmark_move = np.vstack((landmark_move, line))

    print("Landmark move to center Done")
    
    # 標準化 mel spec
    # instantiate StandardScaler
    scaler = MaxAbsScaler()
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    
    # Apply MaxAbs scaling
    landmark_norm = scaler.fit_transform(landmark_move)
    with open("scaler_landmark.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(f"norm_landmark", landmark_norm)


if __name__ == "__main__":
    # pls create this folder manually
    video_path = "C:\\Speech2Face_database_115\\video"
    landmark_path = "C:\\Speech2Face_database_115\\landmark"
    audio_path = "C:\\Speech2Face_database_115\\audio"
    mel_path = "C:\\Speech2Face_database_115\\mel"

    # (step 1) video to audio
    iterFolder(fun=video2audio, src_path=video_path, dest_path=audio_path)

    # (step 2) audio to mel
    sample_idx = iterFolder(fun=mel_spec, src_path=audio_path, dest_path=mel_path)
    # (step 3) video to landmark
    iterFolder(fun=landmark, src_path=video_path, dest_path=landmark_path, frame_idx=sample_idx)

    # # (step 4) merge mel with norm
    merge_mel_with_norm(src_path=mel_path)

    # # (step 5) merge landmark with move to (0,0) and norm
    merge_landmark_move_center_with_norm(src_path=landmark_path)
