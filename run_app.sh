#!/bin/bash

# Activate virtual environment and run Flask app
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please create one first:"
    echo "python3 -m venv venv"
    exit 1
fi

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

cd frontend
python app.py
