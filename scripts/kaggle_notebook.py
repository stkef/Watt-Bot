# WattBot 2026 - Kaggle notebook driver (paste into a notebook cell, or %run it).
# Inputs attached to the notebook:
#   /kaggle/input/wattbot-2026/                -> competition data (train_QA.csv, test_Q.csv, metadata.csv, Score.py)
#   /kaggle/input/wattbot-corpus/cache/parsed/ -> private dataset made once by `wattbot fetch && wattbot parse`
# Secret: GEMINI_API_KEY in Kaggle Secrets.
import os, shutil, subprocess, sys

COMP = "/kaggle/input/wattbot-2026"
CORPUS = "/kaggle/input/wattbot-corpus/cache"
REPO = "https://github.com/stkef/Watt-Bot.git"
TAG = os.environ.get("WATTBOT_TAG", "main")

subprocess.run(["git", "clone", "--depth", "1", "--branch", TAG, REPO, "/kaggle/working/wattbot"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/kaggle/working/wattbot"], check=True)

os.environ["WATTBOT_DATA"] = COMP
os.environ["WATTBOT_CACHE"] = "/kaggle/working/cache"
os.environ["WATTBOT_RUNS"] = "/kaggle/working/runs"
shutil.copytree(CORPUS, "/kaggle/working/cache", dirs_exist_ok=True)

run = lambda *a: subprocess.run(["wattbot", *a], check=True)
run("index")
run("run", "--split", "train", "--name", "nb")
run("score", "/kaggle/working/runs/nb/train_predictions.csv")
run("run", "--split", "test", "--name", "nb")
shutil.copy("/kaggle/working/runs/nb/test_predictions.csv", "/kaggle/working/submission.csv")
