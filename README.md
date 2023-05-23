# Decode me

## Clone the project

```bash
mkdir ~/code/jaleika && cd "$_"
git clone git@github.com:jaleika/decode_me.git
cd project_name
```
## Add a raw_data directory

```bash
mkdir raw_data
```

## Setup a new virtualenv

```bash
pyenv virtualenv decode_me
cd ~/code/jaleika/decode_me
pyenv local decode_me # This command creates a .python-version file in the directory of the project containing the name of the virtual env (cat .python-version). This is what allows pyenv to know which virtual env to use.
```

## Install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt

#### To install the package the first time:
make install clean

## Incase you have to reinstall the package
## use:
make reinstall_package

##In order to run the API:
make run_api

```
