# app/constants.py

# 1. FunASR 模型 ID 映射
MODELS = {
    "paraformer-zh": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}

# 2. 其他业务常量（例如：支持的音频格式、默认分页大小等）
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a"]

DEFAULT_PAGE_SIZE = 20


