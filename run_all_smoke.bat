@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_limit.py --size limit-small
python scripts\prepare_data.py --size limit-small
pip install pyserini
python scripts\run_bm25.py --size limit-small
python scripts\evaluate.py --size limit-small --run results\bm25_limit-small.txt
pip install -r requirements-models.txt
python scripts\run_dense_baseline.py --model contriever --size limit-small
python scripts\run_dense_baseline.py --model medcpt --size limit-small
echo.
echo SMOKE TEST FINISHED. See results folder.
pause
