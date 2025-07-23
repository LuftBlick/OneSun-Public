@echo off
cd /d "%~dp0"
cd apps\main
call conda activate env_dl
streamlit run main.py
pause