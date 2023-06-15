import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import cv2
import pickle
from moviepy.editor import VideoFileClip, AudioFileClip
import mediapipe as mp
import shutil

# Define network
class NetWork(pl.LightningModule):
    def __init__(self, n_in, n_hidden, n_out):
        super(NetWork, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x


def iterFolder(fun, **kwargs):
    src_path = kwargs["src_path"]
    dest_path = kwargs["dest_path"]
    print(src_path, dest_path)

    if "frame_idx" in kwargs:
        frame_idx = kwargs["frame_idx"]
        for idx, file in enumerate(os.listdir(src_path)):
            curr_s_path = f"{src_path}/{file}"
            curr_d_path = f"{dest_path}/{file}"
            fun(curr_s_path, curr_d_path, frame_idx[idx])
        return 0


    if "aud_path" in kwargs:
        aud_path = kwargs["aud_path"]
        for file in os.listdir(src_path):
            curr_s_path = f"{src_path}/{file}"
            curr_d_path = f"{dest_path}/{file}"
            curr_aud_path = f"{aud_path}/{file.split('.')[0]}.wav"
            fun(curr_s_path, curr_d_path, curr_aud_path)
        return 0        

    sample_idx = []
    for idx, file in enumerate(os.listdir(src_path)):
        curr_s_path = f"{src_path}/{file}"
        curr_d_path = f"{dest_path}/{file}"

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

    
    # 讀取StandardScaler物件
    with open("scaler_mel.pkl", "rb") as f:
            zscore = pickle.load(f)

    # 資料標準化
    log_mel = zscore.fit_transform(log_mel)

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
    arr = np.empty((0, 80))
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

                arr = np.vstack((arr, col))

        # reading next frame from video
        success, image = vidcap.read()
        img_idx += 1

        if img_idx == frame_idx:
            break


    # move landmark to (0,0)
    # mean of 78 and 308 is the x, y coordinate of center
    landmark_move = np.empty((0, 80))
    for line in arr:
        # each loop for one image
        # get x, y coordinate
        x_mean = (line[16]+line[52]) / 2
        y_mean = (line[17]+line[53]) / 2

        line[::2] -= x_mean  # x coordinate
        line[1::2] -= y_mean  # y coordinate

        landmark_move = np.vstack((landmark_move, line))

    print("Landmark move to center Done")
    
    # 歸一化 landmark to [-1,1]
    # 讀取 MaxAbs scaler 物件
    with open("scaler_landmark.pkl", "rb") as f:
            maxabs = pickle.load(f)
            
    # Apply MaxAbs
    landmark_norm = maxabs.fit_transform(landmark_move)
        
    np.save(f"{dest_path.split('.')[0]}", landmark_norm)
    
    
def calling_model(mel, landmark,landmark_hat,ckpt):
    # load model
    # 載入 checkpoint
    model = NetWork.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))
    # disable randomness, dropout, etc...
    model.eval()

    for file in os.listdir(mel):
        # Load testing data
        X = np.load(f"{mel}/{file}")
        X = torch.from_numpy(X).float()
        # y = np.load(f"{landmark}/{file}")
        # y = torch.from_numpy(y).float()
        
        with torch.no_grad():
            y_predict = model(X)
            # loss = F.mse_loss(y_predict,y)
            # print("Loss of testing data: ", loss.item())
            np.save(f"{landmark_hat}/{file.split('.')[0]}", y_predict.numpy())




def landmark_to_frame(src_path, dest_path):
    # Create a folder where store all the sub clips' audio
    if not os.path.exists(f"{dest_path.split('.')[0]}"):
        os.mkdir(f"{dest_path.split('.')[0]}")    
        
    weight = 400
    height = 300
    # mid point of FHD plot => (540, 960)
    lines = np.load(src_path)
    # 讀取 MaxAbs scaler 物件
    with open("scaler_landmark.pkl", "rb") as f:
        zscore = pickle.load(f)

    # 資料還原
    lines = zscore.inverse_transform(lines)
    # each loop for one image
    for idx, line in enumerate(lines):
        # create img
        img = np.zeros((height, weight, 3), dtype="uint8")
        img.fill(0)
        # get x, y coordinate
        for coordinate in zip(line[0::2], line[1::2]):
            # print(coordinate[0], coordinate[1])
            relative_x = int(float(coordinate[0]) * 1080)
            relative_y = int(float(coordinate[1]) * 1920)
            relative_x += 200
            relative_y += 150
            # draw feature points
            cv2.circle(img, (relative_x, relative_y),
                        radius=1, color=(255, 255, 255), thickness=5)

        cv2.imwrite(
            f"{dest_path.split('.')[0]}/{idx}.jpg", img)
        
        
def frame_to_video(src_path, dest_path):
    img = cv2.imread(f'{src_path}/0.jpg')
    fps = 60.0
    size = (img.shape[1], img.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(f"{dest_path}.mp4", fourcc, fps, size)

    files = os.listdir(f'{src_path}')
    frame_num = len(files)
    print(f'Number of frames: {frame_num}')
    for i in range(frame_num):
        fileName = f"{src_path}/" + str(i) + '.jpg'
        img = cv2.imread(fileName)
        videoWrite.write(img)
    videoWrite.release()
    
    
def add_sound_track(vid_path, dest_path, aud_path):
    # 讀取影片
    video = VideoFileClip(vid_path)

    # 讀取音頻文件
    audio = AudioFileClip(aud_path)

    # 添加音頻到影片中
    video = video.set_audio(audio)

    # 寫入新文件
    video.write_videofile(f"{dest_path}", fps=video.fps)

    
if __name__ == "__main__":
    # pls create this folder manually
    video_path = "predict/video"
    audio_path = "predict/audio"
    mel_path = "predict/mel"
    landmark_path = "predict/landmark"
    landmark_hat_path = "predict/landmark_hat"
    frame_path = "predict/frame"
    frame_hat_path = "predict/frame_hat"
    video_no_snd = "predict/video_no_snd"
    video_no_snd_hat = "predict/video_no_snd_hat"
    video_with_snd = "predict/video_with_snd"
    video_with_snd_hat = "predict/video_with_snd_hat"
    


    # # (step 1) video to audio
    # iterFolder(fun=video2audio, src_path=video_path, dest_path=audio_path)
    
    # (step 2) audio to mel
    sample_idx = iterFolder(fun=mel_spec, src_path=audio_path, dest_path=mel_path)
    # # (step 3) video to landmark
    # iterFolder(fun=landmark, src_path=video_path, dest_path=landmark_path, frame_idx=sample_idx)

    ## (step 4)
    ckpt_path = r"expt_epoch\version_0\checkpoints\epoch=9-step=10.ckpt"
    calling_model(mel=mel_path, landmark=landmark_path,landmark_hat=landmark_hat_path,ckpt=ckpt_path)

    # # (step 5)
    # # This for origin video
    # iterFolder(fun=landmark_to_frame, src_path=landmark_path, dest_path=frame_path)
    # This for predict video
    iterFolder(fun=landmark_to_frame, src_path=landmark_hat_path, dest_path=frame_hat_path)
    
    # # (step 6)
    # # iterFolder(fun=frame_to_video, src_path=frame_path, dest_path=video_no_snd)
    iterFolder(fun=frame_to_video, src_path=frame_hat_path, dest_path=video_no_snd_hat)
    
    # # (step 7)
    # # iterFolder(fun=add_sound_track, src_path=video_no_snd, dest_path=video_with_snd, aud_path=audio_path)
    iterFolder(fun=add_sound_track, src_path=video_no_snd_hat, dest_path=video_with_snd_hat, aud_path=audio_path)
    
    
    
    
    # # 這個是刪除 predict 和 predict 下所有的東西
    # shutil.rmtree("predict")
    
    # # 這個可以一次創建所有需要的資料夾
    # # 然後把要 testing 的影片放到 video 裡
    # folderList = ['video', 'audio', 'mel', 'landmark', 'landmark_hat', 'frame', 'frame_hat', 'video_no_snd', 'video_no_snd_hat', 'video_with_snd', 'video_with_snd_hat']
    # for folder in folderList: 
    #     os.makedirs(f"predict/{folder}")