"""
Telephony Drivers Layer for AI Voice Calling Assistant.
Exports BaseTelephonyDriver, WindowsAudioDriver, VirtualSimulatorDriver, and LinuxDriver.
"""

from telephony.base_driver import BaseTelephonyDriver
from telephony.windows_driver import WindowsAudioDriver
from telephony.virtual_driver import VirtualSimulatorDriver
from telephony.linux_driver import LinuxDriver

__all__ = [
    "BaseTelephonyDriver",
    "WindowsAudioDriver",
    "VirtualSimulatorDriver",
    "LinuxDriver",
]
