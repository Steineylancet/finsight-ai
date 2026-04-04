#!/bin/bash
# Azure App Service startup script
cd /home/site/wwwroot
uvicorn backend.main:app --host 0.0.0.0 --port 8000
