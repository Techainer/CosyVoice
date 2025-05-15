import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio # type: ignore

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B-ft', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('samples/cdteam.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # save zero_shot spk for future usage
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

print(cosyvoice.save_spkinfo())

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('bistream usage, you can use generator as input, this is useful when using text llm model as input', prompt_speech_16k, stream=False)):
for i, j in enumerate(cosyvoice.inference_cross_lingual('dự kiến tăng mức phụ cấp [laughter] ưu đãi nghề cho giáo viên mầm non từ bốn mươi lăm phần trăm đến tám mươi phần trăm [laughter].', prompt_speech_16k, stream=False)):
    torchaudio.save('inference_cross_lingual_8epoch.wav', j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('Dự kiến tăng mức phụ cấp ưu đãi nghề cho giáo viên mầm non từ 45% đến 80%', 'female voice', prompt_speech_16k, stream=False)):
#     torchaudio.save('inference_instruct2_4epoch.wav', j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
    
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)