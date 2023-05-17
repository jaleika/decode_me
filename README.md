# Decode me

## Clone the project

```bash
mkdir ~/code/<PROJECT_LEADER_GITHUB_NICKNAME> && cd "$_"
git clone git@github.com:<PROJECT_LEADER_GITHUB_NICKNAME>/<PROJECT_NAME>.git
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
```
