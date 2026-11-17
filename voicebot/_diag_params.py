from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransportParams

a = TransportParams(audio_in_enabled=True)
b = LocalAudioTransportParams(audio_out_enabled=True, output_device_index=5)
for name, p in (("input TransportParams", a), ("output LocalAudioTransportParams", b)):
    print(name)
    for f in ("audio_in_enabled", "audio_out_enabled", "audio_in_passthrough",
              "audio_in_sample_rate", "audio_out_sample_rate"):
        print(f"   {f} = {getattr(p, f, '<none>')}")
