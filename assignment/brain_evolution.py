import datetime
from pathlib import Path
import ray

from brain_utils import evolution_brain
from plot import make_plot


CWD = Path.cwd()
FOLDER = CWD / "femke_bot"
FOLDER.mkdir(exist_ok=True)

brain_gens = 2
graph_filename = f"{FOLDER}/femke_bot.json"

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
policy_filename = "femke_bot_brainevolution_100gens_{timestamp}"

evolution_brain(policy_filename, graph_filename, folder="femke_bot", generations=brain_gens)
make_plot("femke_bot/femke_bot_brainevolution_100gens.csv", "best_eval", "femke_bot/femke_bot_brainevolution_100gens_plot.pdf")
ray.shutdown()