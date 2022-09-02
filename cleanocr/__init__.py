import json
import subprocess
from pathlib import Path


def main():
    venv_path = subprocess.check_output("poetry env info --path".split())
    venv_path = venv_path.decode("UTF-8").strip()

    settings = dict()

    Path(".vscode").mkdir(parents=True, exist_ok=True)
    Path(".vscode/settings.json").touch()

    with open(".vscode/settings.json", "r") as f:
        try:
            settings = json.load(f)
        except:
            print("file empty")
        settings["python.pythonPath"] = venv_path

    with open(".vscode/settings.json", "w") as f:
        json.dump(settings, f, sort_keys=True, indent=4)

    print(json.dumps(settings, sort_keys=True, indent=4))
