import pyaudio
pa = pyaudio.PyAudio()
print('=== CABLE INPUT OUTPUT DEVICES (PyAudio) ===')
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    host = pa.get_host_api_info_by_index(int(info['hostApi']))['name']
    name = str(info['name'])
    if ('CABLE Input' in name or 'CABLE In' in name) and info['maxOutputChannels'] > 0:
        print(f"  idx={i:>3} {host:<12} rate={int(info['defaultSampleRate'])}  {name}")
print('=== ALL HOST APIS ===')
for i in range(pa.get_host_api_count()):
    print(' ', pa.get_host_api_info_by_index(i)['name'])
pa.terminate()
