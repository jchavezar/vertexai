from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerationResponse
from proto.marshal.collections import repeated
from proto.marshal.collections import maps

LOCATION = "us-central1"
PROJECT_ID = 'vtxdemos'  # PUT YOUR PROJECT HERE!

aiplatform.init(project=PROJECT_ID, location=LOCATION)


# noinspection PyTypeChecker
def recurse_proto_repeated_composite(repeated_object):
    repeated_list = []
    for item in repeated_object:
        if isinstance(item, repeated.RepeatedComposite):
            item = recurse_proto_repeated_composite(item)
            repeated_list.append(item)
        elif isinstance(item, maps.MapComposite):
            item = recurse_proto_marshal_to_dict(item)
            repeated_list.append(item)
        else:
            repeated_list.append(item)

    return repeated_list


def recurse_proto_marshal_to_dict(marshal_object):
    new_dict = {}
    for k, v in marshal_object.items():
        if not v:
            continue
        elif isinstance(v, maps.MapComposite):
            # noinspection PyTypeChecker
            v = recurse_proto_marshal_to_dict(v)
        elif isinstance(v, repeated.RepeatedComposite):
            # noinspection PyTypeChecker
            v = recurse_proto_repeated_composite(v)
        new_dict[k] = v

    return new_dict


def get_text(response: GenerationResponse):
    """Returns the Text from the Generation Response object."""
    part = response.candidates[0].content.parts[0]
    # noinspection PyBroadException
    try:
        text = part.text
    except:
        text = None

    return text


def get_function_name(response: GenerationResponse):
    return response.candidates[0].content.parts[0].function_call.name


def get_function_args(response: GenerationResponse) -> dict:
    return recurse_proto_marshal_to_dict(response.candidates[0].content.parts[0].function_call.args)
