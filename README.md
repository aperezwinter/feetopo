# feetopo

## Package Installation
The necessary packages are listed in the *requirements.txt* file.

- Create the virtual environment: `python3.10 -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Upgrade pip: `python3.10 -m pip install --upgrade pip`
- Install packages from *requirements.txt*: `pip install -r requirements.txt`
- Uninstall all packages: `pip freeze | xargs pip uninstall -y`
- Deactivate the virtual environment: `deactivate`
- Remove the virtual environment (optional): `rm -rf venv`