$ErrorActionPreference = "Stop"

Set-Location "D:\github\llm4drd-platform"

& "D:\Users\imzho\miniconda3\python.exe" "D:\github\llm4drd-platform\run_server.py" *>> "D:\github\llm4drd-platform\server.log"
