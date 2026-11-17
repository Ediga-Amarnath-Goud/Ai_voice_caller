# Repair the Bluetooth hands-free audio endpoints for the phone.
#
# The phone hands call audio to Windows, Windows exposes no usable endpoint, the
# phone gives up and falls back to its own earpiece. This rebuilds the endpoint:
# evict any process that could be holding it, restart the audio engine, then
# disable/enable the hands-free device so the driver re-registers its KS filter.
#
# Needs administrator. Run:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\repair_bt_audio.ps1

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'repair_bt.log'
Start-Transcript -Path $log -Force | Out-Null

function Step([string]$text) { Write-Host "`n== $text" }

Step 'privileges'
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
Write-Host ("elevated: " + $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

Step 'stopping Phone Link (restarts on its own)'
foreach ($name in @(
    'PhoneExperienceHost',
    'CrossDeviceService',
    'CrossDeviceResume',
    'YourPhoneAppProxyHost'
)) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  stop $($_.ProcessName) $($_.Id)"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

Step 'restarting the audio engine'
Restart-Service AudioEndpointBuilder -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Restart-Service Audiosrv -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Get-Service Audiosrv, AudioEndpointBuilder |
    Select-Object Name, Status |
    Format-Table -AutoSize |
    Out-String |
    Write-Host

Step 'cycling the hands-free device'
$devices = Get-PnpDevice -ErrorAction SilentlyContinue |
    Where-Object { $_.InstanceId -match 'BTHHFENUM' -and $_.FriendlyName -match 'iQOO' }

if (-not $devices) {
    Write-Host '  no iQOO hands-free device found -- is the phone connected?'
}

foreach ($device in $devices) {
    Write-Host "  $($device.FriendlyName)"
    Write-Host "    id     $($device.InstanceId)"
    Write-Host "    status $($device.Status) / problem $($device.Problem)"

    try {
        Disable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false -ErrorAction Stop
        Write-Host '    disabled'
        Start-Sleep -Seconds 2
    } catch {
        Write-Host "    disable failed: $($_.Exception.Message)"
    } finally {
        try {
            Enable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false -ErrorAction Stop
            Write-Host '    enabled'
        } catch {
            Write-Host "    ENABLE FAILED: $($_.Exception.Message)"
            Write-Host '    re-enable it manually in Device Manager (Sound, video and game controllers)'
        }
    }
    Start-Sleep -Seconds 2
}

Step 'hands-free devices after repair'
Get-PnpDevice -ErrorAction SilentlyContinue |
    Where-Object { $_.InstanceId -match 'BTHHFENUM' } |
    Select-Object Status, Problem, FriendlyName |
    Format-Table -AutoSize |
    Out-String |
    Write-Host

Step 'done -- close this window'
Write-Host "log: $log"
Stop-Transcript | Out-Null
