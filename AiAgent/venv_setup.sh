#/bin/env /bin/bash

sudo apt install python3.12-venv -y
python3 -m venv .venv
source .venv/bin/activate
exec bash --rcfile <(echo "source ~/.bashrc; source .venv/bin/activate")
