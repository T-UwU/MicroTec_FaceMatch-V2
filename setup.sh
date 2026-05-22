#!/bin/bash
set -e

echo "Configurando Face Anti-Spoofing API..."

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Instalación completa."
echo "Para iniciar:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
