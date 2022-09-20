
from glob import glob
import argparse, os, os.path as osp
parser = argparse.ArgumentParser()

parser.add_argument('shfile')
parser.add_argument('globpattern')
parser.add_argument('totalgpu')

args = parser.parse_args()

# inputs="/data/DMS_Behavior_Detection/RawVideos/Action_Eating/*/*.mp4"
cmds = []
for i, path in enumerate(glob(args.globpattern)):
    opath = path.replace('.json', '_2.json')
    if not osp.exists(opath):
        cmd = f"{args.shfile} {path} {opath}"
        cmds.append(cmd)
# import ipdb; ipdb.set_trace()
wi = 0

num_cmd_per_window = len(cmds) // 16
print(f'{num_cmd_per_window=}')
for i in range(0, len(cmds), num_cmd_per_window):
    
    _cmds = cmds[i:i+num_cmd_per_window]

    _cmds = "\n".join(_cmds)
    
    
    tmpsh = f'/tmp/script-{wi}.sh'
    with open(tmpsh, 'w') as f:
        f.write(_cmds)
    gpu = wi%8
    if i == 0:
        target_tmux = "run-0"
        tmuxcmd = f"tmux new -s '{target_tmux}' -d 'CUDA_VISIBLE_DEVICES={gpu} sh {tmpsh} || echo Done && sleep 10'"
    else:
        tmuxcmd = f"tmux new-window -n w{wi} -t {target_tmux}: 'CUDA_VISIBLE_DEVICES={gpu} sh {tmpsh} || echo Done && sleep 10'"
    wi += 1
    print(tmuxcmd)
    # os.system(tmuxcmd)