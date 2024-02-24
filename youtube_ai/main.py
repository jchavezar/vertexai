# %%
import glob
import streamlit as st
from pytube import YouTube
from moviepy.editor import *
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


class VideoLLM:
    def __init__(self):
        self.vid_directory = "videos_d"  # video downloads location
        self.aud_directory = "audio_d"  # video downloads location
        self.project_id = "vtxdemos"
        self.bucket_id = "vtxdemos-nba-vid"
        self.gcs_output_path_speech_to_text = "gs://vtxdemos-nbademos/transcription_output"
        self.storage_client = storage.Client(project=self.project_id)
        self.storage_bucket = self.storage_client.get_bucket(self.bucket_id)
        self.client = SpeechClient()

    # Definitions
    def download_video(self, url):
        YouTube(
           url,
           use_oauth=False,
           allow_oauth_cache=True
        ).streams.first().download(self.vid_directory)

        for filename in glob.glob(f"{self.vid_directory}/*.mp4"):
            new_name = "_".join(filename.split(" "))
            os.rename(filename, new_name)

    # %%
    # Google Cloud Storage
    # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

    def speech_to_text(self):
        for video_name in os.listdir(self.vid_directory):
            # Storing the Video in GCS
            self.storage_bucket.blob(self.vid_directory + "/" + video_name).upload_from_filename(self.vid_directory + "/" + video_name)

            # Speech to Text (Transform to mp3 and Upload to GCS)
            vid = VideoFileClip(f"{self.vid_directory}/{video_name}")
            audio_name = f"{video_name.split('.')[-2]}.mp3"
            file_path = f"{self.aud_directory}/{audio_name}"
            vid.audio.write_audiofile(file_path)
            self.storage_bucket.blob(file_path).upload_from_filename(file_path)

            gcs_uris = [f"gs://{self.bucket_id}/" + blob.name for blob in self.storage_bucket.list_blobs(prefix="audio")]

            files = [
                cloud_speech.BatchRecognizeFileMetadata(uri=uri)
                for uri in gcs_uris
            ]
            config = cloud_speech.RecognitionConfig({
                "auto_decoding_config": cloud_speech.AutoDetectDecodingConfig(),
                "language_codes": ["en-US"],
                "model": "long",
            })

            request = cloud_speech.BatchRecognizeRequest(
                {"recognizer" : f"projects/{self.project_id}/locations/global/recognizers/_",
                 "config": config,
                 "files": files,
                 "recognition_output_config": cloud_speech.RecognitionOutputConfig({
                     "gcs_output_config": cloud_speech.GcsOutputConfig({
                         "uri": self.gcs_output_path_speech_to_text,
                     }),
                 }),
                 }
            )

            operation = self.client.batch_recognize(request=request)
            print("Waiting for operation to complete...")
            response = operation.result(timeout=1000)


def refresh_state():
    st.session_state['status'] = 'submitted'


def main():
    st.title("Video to Text")
    url = st.text_input('Enter your YouTube video link', 'https://youtu.be/dccdadl90vs', on_change=refresh_state)
    if url:
        with st.spinner("Loading Video... Please Wait"):
            v = VideoLLM()
            v.download_video(url)
        with st.spinner("Transforming Video to Audio and to Text"):
            v.speech_to_text()
    else:
        pass


if __name__ == "__main__":
    main()
