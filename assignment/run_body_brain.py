import torch

from brain_utils import detect_io_sizes, Policy, experiment_brain_one_terrain


FOLDER = "femke_bot"
graph_filename = f"{FOLDER}/femke_bot.json"
brain_filename = f"{FOLDER}/femke_bot_brainevolution_100gens.pth"

input_size, output_size = detect_io_sizes(graph_filename)
showbrain = Policy(input_size=13, output_size=8)
showbrain.load_state_dict(torch.load(brain_filename))
experiment_brain_one_terrain(showbrain, graph_filename, duration=2, view=True, folder=FOLDER)