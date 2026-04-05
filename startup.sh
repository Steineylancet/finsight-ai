#!/bin/bash
# Azure App Service startup script
cd /home/site/wwwroot
pip install -r requirements.txt --quiet
uvicorn backend.main:app --host 0.0.0.0 --port 8000
