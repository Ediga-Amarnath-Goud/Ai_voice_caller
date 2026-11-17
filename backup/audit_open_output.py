import pyaudio, numpy as np
pa = pyaudio.PyAudio()
for idx, rate in ((14, 48000), (5, 44100)):
    info = pa.get_device_info_by_index(idx)
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                         output=True, output_device_index=idx,
                         frames_per_buffer=960)
        tone = (np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 8000).astype('<i2')
        stream.write(tone[:960].tobytes(), 10)
        stream.stop_stream(); stream.close()
        print(f"OPEN OK  idx={idx} {info['hostApi']} rate={rate}  {info['name']}")
    except Exception as exc:
        print(f"OPEN FAIL idx={idx} rate={rate}: {exc}")
pa.terminate()
