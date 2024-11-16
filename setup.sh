#!/bin/bash
python3 -m venv .
. bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 setup.py install