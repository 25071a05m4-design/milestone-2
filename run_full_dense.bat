@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\run_dense_baseline.py --model contriever --size limit
python scripts\run_dense_baseline.py --model medcpt --size limit
pause
