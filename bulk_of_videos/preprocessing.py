# %%
# Libraries Definitions
import re
import json
from moviepy.editor import *
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.storage import transfer_manager
from google.cloud.speech_v2.types import cloud_speech

project_id = "vtxdemos"
bucket_id = "vtxdemos-nbademos"
video_dir_path = "../videos"
audio_dir_path = "../audio"
gcs_audio = "gs://vtxdemos-nbademos/audio"
gcs_output_path_speech_to_text = "gs://vtxdemos-nbademos/transcription"

client = storage.Client(project_id)
bucket = client.bucket(bucket_id)
names = [f"gs://{bucket_id}/" + blob.name for blob in bucket.list_blobs()]
blob = bucket.blob("clippers_thunders.mp4")

client = SpeechClient()
# noinspection PyTypeChecker
config = cloud_speech.RecognitionConfig(
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    model="long",
)

# %%
# Download Videos Locally
blob_names = [blob.name for blob in bucket.list_blobs()]
results = transfer_manager.download_many_to_path(
    bucket, blob_names, destination_directory=video_dir_path, max_workers=8
)

# %%
# From Video to Audio
for n, blob in enumerate(bucket.list_blobs(prefix="videos")):
    if n == 0:
        continue
    video = blob.name
    vide_name = video.split("/")[-1]
    print(video)
    vid = VideoFileClip(f"{video_dir_path}/{vide_name}")
    vid.audio.write_audiofile(f"{audio_dir_path}/{vide_name.split('.')[-2]}.mp3")

# %%
# Upload Audio to Google Cloud Storage
mp3_pattern = re.compile(r'\.mp3$', re.IGNORECASE)
audio_files = [audio_dir_path + "/" + filename for filename in os.listdir(audio_dir_path) if
               mp3_pattern.search(filename)]
transfer_manager.upload_many_from_filenames(bucket=bucket, filenames=audio_files)

# %%
# Transcriptions using Chirp
gcs_uris = [f"gs://{bucket_id}/" + blob.name for blob in bucket.list_blobs(prefix="audio")]

files = [
    cloud_speech.BatchRecognizeFileMetadata(uri=uri)
    for uri in gcs_uris
]
# noinspection PyTypeChecker
request = cloud_speech.BatchRecognizeRequest(
    recognizer=f"projects/{project_id}/locations/global/recognizers/_",
    config=config,
    files=files,
    recognition_output_config=cloud_speech.RecognitionOutputConfig(
        gcs_output_config=cloud_speech.GcsOutputConfig(
            uri=gcs_output_path_speech_to_text,
        ),
    ),
)
operation = client.batch_recognize(request=request)
print("Waiting for operation to complete...")
response = operation.result(timeout=1000)

# %%
# Exploring Transcriptions
audio_trans = [blob.name for blob in bucket.list_blobs(prefix="transcription")]

str_json = bucket.blob(audio_trans[0]).download_as_text()

dict_res = json.loads(str_json)

for result in dict_res["results"]:
    for i in result["alternatives"]:
        print(i["transcript"])
