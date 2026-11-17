import pyaudio
pa = pyaudio.PyAudio()
print('=== PYAUDIO INPUT DEVICES ===')
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    host = pa.get_host_api_info_by_index(int(info['hostApi']))['name']
    if info['maxInputChannels'] > 0:
        print(f'  idx={i:>3} {host:<12} {info["name"]}')
pa.terminate()