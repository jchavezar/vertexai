#%%
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


# noinspection PyTypeChecker
def transcribe_batch_gcs_input_inline_output_v2(
        project_id: str,
        gcs_uri: str,
) -> cloud_speech.BatchRecognizeResults:
    """Transcribes audio from a Google Cloud Storage URI.

    Args:
        project_id: The Google Cloud project ID.
        gcs_uri: The Google Cloud Storage URI.

    Returns:
        The RecognizeResponse.
    """
    # Instantiates a client
    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="long",
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    # Transcribes the audio into text
    operation = client.batch_recognize(request=request)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=800)

    for result in response.results[gcs_uri].transcript.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response.results[gcs_uri].transcript


#%%
text = transcribe_batch_gcs_input_inline_output_v2("vtxdemos", "gs://vtxdemos-datasets-private/audio/all_star_nba.wav")
