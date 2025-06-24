import wave
import io

def pcm_to_wav(pcm_data, bit_depth=16, num_channels=1, sample_rate=16000):
    with io.BytesIO() as file:
        a = wave.Wave_write(file)
        a.setparams((num_channels, bit_depth//8, sample_rate, 0, 'NONE', 'NONE'))
        a.writeframes(pcm_data)
        a.close()
        wav_data = file.getvalue()
    
    return wav_data