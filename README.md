# QWOP-RL

### Dependencies
- Python 3.7
- PyTorch
- Numpy
- Selenium (https://pypi.org/project/selenium/)
- pytesseract (https://pypi.org/project/pytesseract/)
- pynput (https://pypi.org/project/pynput/)
- mss (https://pypi.org/project/mss/)

### How to Run
Ensure the following files are in the local folder:
- `assets/assetbundle.parcel`
- `lib/howler.js`
- `Athletics.html`
- `main.css`
- `QWOP1.min.js`
- `chromedriver.rb`
- `agent.py`
- `environment.py`
- `runner.py`
- `qwop-rl.py`
- `settings.yaml`

#### Install Chromedriver (macOS homebrew file provided)
````
brew install chromdriver.rb 
````

#### CLI
All settings and parameters are loaded via a yaml file. A starter file is provided.
```
python qwop-rl.py [settings.yaml] [options]

usage: qwop-rl.py [-h] [-e E] [-m M] [-l L] [-c C] [-f F] [-q] yamlParams

positional arguments:
  yamlParams  Settings & parameters yaml file path.

optional arguments:
  -h, --help  show this help message and exit
  -e E        Set number of episodes to run
  -m M        Set file for loading a pre-existing model.
  -l L        Set output log file name.
  -c C        Set checkpoint file name.
  -f F        Set checkpoint save frequency.
  -q          Quiet run. Non-verbose printout of episode details.
```