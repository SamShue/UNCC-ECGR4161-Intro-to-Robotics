# ECGR 4161 Intro to Robotics Notebooks

This repository contains notebook assignments for motor control simulations:

- `h_bridge_motor_driver_assignment.ipynb`
- `stepper_motor_full_step_assignment.ipynb`

## Environment Setup

### Linux or macOS (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ecgr4161-venv --display-name "Python (ecgr4161-venv)"
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ecgr4161-venv --display-name "Python (ecgr4161-venv)"
```

## Running the Notebooks

1. Activate the virtual environment.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Open one of the `.ipynb` files.
4. Select kernel **Python (ecgr4161-venv)** if prompted.

## Notes

- The `.venv/` directory is intentionally ignored in git.
- If widgets do not render in notebook outputs, make sure `ipywidgets` is installed in the active environment.
