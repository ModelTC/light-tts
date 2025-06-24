import base64
import uuid
import numpy as np
import asyncio
from light_tts.utils.param_utils import check_request
from light_tts.server.httpserver.manager import HttpServerManager
from fastapi.responses import Response
from light_tts.utils.log_utils import init_logger
from light_tts.static_config import dict_language

logger = init_logger(__name__)
   

async def health_check(httpserver_manager: HttpServerManager, g_id_gen, lora_styles):
    try:
        gen_ans_list = []
        for style in lora_styles:
            for language in set(dict_language.values()):
                request_dict = {
                    "tts_model_name": style,
                    "text": "good",
                    "text_id": str(uuid.uuid4()),
                    "text_language": language,
                    "ref_free": False,
                    "return_format": "json",
                }         
                check_request(request_dict, lora_styles)

                request_id = g_id_gen.generate_id()
                results_generator = httpserver_manager.generate(request_dict, request_id)
                # # Non-streaming case
                gen_ans_list.append(results_generator)
        
        all_ans = await asyncio.gather(*gen_ans_list)
        return True
    except Exception as e:
        logger.critical("health_check error:", e)
        return False