@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts\download_limit.py --size limit
python scripts\prepare_data.py --size limit
python scripts\run_bm25.py --size limit
python scripts\evaluate.py --size limit --run results\bm25_limit.txt
pause
