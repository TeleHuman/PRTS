This README is adapted from the [OpenPI LIBERO setup guide](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md).

The steps below are for running LIBERO evaluation

## Terminal 1: Environment Setup + LIBERO Simulation

```bash
# 1) Create and activate the environment
conda create -n libero python=3.8 -y
conda activate libero

# 2) Clone LIBERO and pin to the expected commit
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
cd ..

# 3) Install dependencies
pip install -r requirements.txt -r LIBERO/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113

# 4) Install editable packages
# Replace <PRTS_ROOT> with your local PRTS root path
uv pip install -e <PRTS_ROOT>/third_party/openpi-client
uv pip install -e LIBERO

# 5) Make LIBERO importable
export PYTHONPATH="$PYTHONPATH:$PWD/LIBERO"

# 6) Run the simulation
python main.py

# If you hit EGL/MuJoCo rendering issues
MUJOCO_GL=glx python main.py
```

## Terminal 2: Start Policy Server

Run this command from the PRTS repository root:

```bash
python scripts/serve_policy.py --env LIBERO
```