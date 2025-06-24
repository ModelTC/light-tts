from schema import Schema, And, Use, Optional, Or, SchemaError
from light_tts.static_config import dict_language

def check_request(request_dict):
    config_schema = Schema({
        # "tts_model_name": And(str, lambda x: x in lora_styles),
        "text": And(str, len),
        "prompt_wav_path": "",
        # Optional("text_id"):Or(None, And(str, len)),
        # "text_language": And(str, lambda x : x in dict_language.keys()),
        # "ref_free":And(bool, lambda x: x == False),
        "return_format": And(str, lambda x: x in ["json", "wav_stream", "wav_local_file"])
    })
    config_schema.validate(request_dict)
    return

