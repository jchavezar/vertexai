# %%
import glob
import json
import time
import vertexai
import gem_utils
import streamlit as st
from pytube import YouTube
from moviepy.editor import *
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel, Tool

vertexai.init(project="vtxdemos", location="us-central1")



class VideoLLM:
    def __init__(self):
        self.vid_directory = "videos_d"  # video downloads location
        self.aud_directory = "audio_d"  # video downloads location
        self.project_id = "vtxdemos"
        self.bucket_id = "vtxdemos-nba-vid"
        self.gcs_output_path_speech_to_text = "gs://vtxdemos-nba-vid/transcription_output"
        self.storage_client = storage.Client(project=self.project_id)
        self.storage_bucket = self.storage_client.get_bucket(self.bucket_id)
        self.client = SpeechClient()
        self.model = GenerativeModel("gemini-1.0-pro-001")
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
        # Getting tables from stats
        # gemini-pro functions
        self.get_stats_funct = {
            "name": "get_stats_funct",
            "description": "getting nba statistics from teams",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_1": {
                        "type": "string",
                        "description": "an nba team"
                    },
                    "team_2": {
                        "type": "string",
                        "description": "an nba team"
                    },
                },
                "required": [
                    "team_1",
                    "team_2"
                ]
            }
        }

        self.all_tools = Tool.from_dict(
            {
                "function_declarations": [
                    self.get_stats_funct,
                ]
            }
        )
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


    def get_stats(self, team_1: str, team_2: str):
        import requests
        from bs4 import BeautifulSoup

        utils_dictionary = {
            "Boston Celtics": "celtics",
            "Cleveland Cavaliers": "cavaliers",
            "Denver Nuggets": "nuggets",
            "Brooklyn Nets": "nets",
            "Charlotte Hornets": "hornets",
            "Chicago Bulls": "bulls",
            "Detroit Pistons": "pistons",
            "Indiana Pacers": "pacers",
            "Minnesota Timberwolves": "timberwolves",
            "Atlanta Hawks": "hawks",
            "Milwaukee Bucks": "bucks",
            "New York Knicks": "knicks",
            "Golden State Warriors": "warriors",
            "Dallas Mavericks": "mavericks",
            "L a clippers": "clippers",
            "Houston Rockets": "rockets",
            "Los Angeles Lakers": "lakers",
            "Miami Heat": "heat",
            "Philadelphia 76ers": "76ers",
            "Memphis Grizzlies": "grizzlies",
            "New Orleans Pelicans": "pelicans",
            "Oklahoma City Thunder": "thunder",
            "Portland Trail Blazers": "blazers",
            "Toronto Raptors": "raptors",
        }

        responses = self.model.generate_content(f"""
        You are a NBA analyst, your task is to get the intent/name of the `Teams` and extract the pseudonym `Value` 
        from the dictionary below:
        
        Dictionary
        str{utils_dictionary}
        
        ###Example###
        Teams:
        team_1: Chicago super bulls
        team_2: New York Sun Nicks
        
        Output: bulls, nicks
        ###End of Example###
        
        Teams:
        team_1: {team_1}
        team_2: {team_2}
        
        Output:
        """)
        team_1, team_2 = responses.text.split(",")

        url = f"https://www.statmuse.com/nba/ask/{team_1}-vs-{team_2}-stats-last-5-games"
        req = requests.get(url)
        sp = BeautifulSoup(req.content, 'lxml')
        table = sp.find("table")
        table_body = table.find("tbody")
        rows = table_body.find_all("tr")
        data_from_table = []

        # Cleaning Data
        for row in rows:
            cols = row.find_all("td")
            cols = [ele.text.strip() for ele in cols]
            data_from_table.append([ele for ele in cols if ele])

        stats = {}
        for num, text in enumerate(data_from_table):
            stats[f"game_on_date_{text[1]}"] = {
                "team": text[0],
                "date": text[1],
                "match": " ".join(text[2:5]),
                "result": text[5],
                "min": text[6],
                "points": text[7],
                "reb": text[8],
                "assist": text[9],
                "steals": text[10],
                "blocked_shots": text[11],
                "turnovers": text[12],
                "field_goals_made": text[13],
                "field_goals_attempted": text[14],
                "field_goals_%": text[15],
                "3_point_field_goals_made": text[16],
                "3_point_field_goals_attempted": text[17],
                "3_point_field_goals_attempted_%": text[18],
                "free_throws_made": text[19],
                "free_throws_attempted": text[20],
                "free_throws_%": text[21],
                "power_forward": text[22],
            }
            st.write(stats)
        return str(stats)
    # %%


    def generate(self, transcript: str) -> str:
        input_text = "Give me the last statistics for the match between the marvelous Celtics and Bulls"
        llm_model = GenerativeModel("gemini-pro")
        fc_chat = llm_model.start_chat()
        res = fc_chat.send_message(
            input_text,
            tools=[self.all_tools],
            safety_settings=self.safety_settings
        )

        text = gem_utils.get_text(res)
        if not text:
            name = gem_utils.get_function_name(res)
            args = gem_utils.get_function_args(res)
            print(f"AGENT: FUNCTION CALL: {name}({args})\n")
            if name == "get_stats_funct":
                stats = self.get_stats(args["team_1"], args["team_2"])
            else:
                stats = "There is no information yet."

            llm_res = fc_chat.send_message(
                f"""You are an expert in NBA analysis, oftentimes funny!, these are your instructions:
                - Your first task is to Extract important highlights from the `Transcript`.
                - The following `Statistics` are from the last 5 games between both teams, your second task is to analyze 
                and enrich your comments to be creative.
                - With all the information create a very accurate and detailed analysis of the game.
                - Do not fake your answer, you have many elements to create a very factual description.
                
                Transcript:
                {transcript}
                
                Statistics:
                {stats}
                
                Output:
                """,
                generation_config={"max_output_tokens": 8000}
            )
            print(llm_res.text)

            return llm_res.text


def refresh_state():
    st.session_state['status'] = 'submitted'

def main():
    bucket = storage.Client(project="vtxdemos").bucket("vtxdemos-nba-vid")

    st.title("Video to Text")
    #url = st.text_input('Enter your YouTube video link', on_change=refresh_state)
    url = "https://www.youtube.com/watch?v=AdIhsl7Joms"
    st.video(url)
    if url:
        with st.spinner("Loading Video... Please Wait"):
            start_time = time.time()
            v = VideoLLM()
            v.download_video(url)
            st.write(f":green[Loading Video and Preprocessing Time: ]{time.time() - start_time}")
        with st.spinner("Transforming Video to Audio and to Text"):
            start_time = time.time()
            v.speech_to_text()
            st.write(f":green[Transformation Processing Time: ]{time.time() - start_time}")
        audio_trans = [blob.name for blob in bucket.list_blobs(prefix="transcription")]
        str_json = bucket.blob(audio_trans[0]).download_as_text()
        dict_res = json.loads(str_json)
        _text = []
        for result in dict_res["results"]:
            for i in result["alternatives"]:
                _text.append(i["transcript"])
        st.write(_text)
        st.info(v.generate(" ".join(_text)))

    else:
        pass

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
