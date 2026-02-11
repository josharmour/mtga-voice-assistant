# Kill hung MTGA Voice Advisor processes
Get-Process | Where-Object {$_.ProcessName -eq "python" -or $_.ProcessName -eq "pythonw"} | ForEach-Object {
    $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    if ($cmdline -like "*main.py*" -or $cmdline -like "*mtga*advisor*") {
        Write-Host "Killing process $($_.Id): $($_.ProcessName) - $cmdline"
        Stop-Process -Id $_.Id -Force
    }
}
Write-Host "Done. All MTGA Advisor processes killed."
