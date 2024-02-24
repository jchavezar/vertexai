#%%
import json
import vertexai
import gem_utils
import streamlit as st
from google.cloud import storage
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
            print(stats)
        return str(stats)

    def generate(self, transcript: str) -> str:
        input_text = "Give me the last statistics for the match between the mavericks and cavaliers"
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


def main():
    v = VideoLLM()
    bucket = storage.Client(project="vtxdemos").bucket("vtxdemos-nba-vid")
    print("Video to Text")
    url = "https://www.youtube.com/watch?v=AdIhsl7Joms"
    st.video(url)
    if url:
        audio_trans = [blob.name for blob in bucket.list_blobs(prefix="transcription")]
        str_json = bucket.blob(audio_trans[0]).download_as_text()
        dict_res = json.loads(str_json)
        _text = []
        for result in dict_res["results"]:
            try:
                for i in result["alternatives"]:
                    _text.append(i["transcript"])
            except:
                pass

        st.write(_text)
        st.info(v.generate(" ".join(_text)))

    else:
        pass


if __name__ == "__main__":
    main()
