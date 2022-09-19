
from glob import glob
import argparse, os
parser = argparse.ArgumentParser()

parser.add_argument('shfile')
parser.add_argument('globpattern')
parser.add_argument('totalgpu')

args = parser.parse_args()

# inputs="/data/DMS_Behavior_Detection/RawVideos/Action_Eating/*/*.mp4"
cmds = []
for i, path in enumerate(glob(args.globpattern)):
    g = i%int(args.totalgpu)
    cmd = f"{args.shfile} {g} {path}"
    cmds.append(cmd)
wi = 0
for i in range(0, len(cmds), int(args.totalgpu)):
    
    _cmds = cmds[i:i+int(args.totalgpu)]

    _cmds = "\n".join(_cmds)
    
    
    tmpsh = f'/tmp/script-{wi}.sh'
    with open(tmpsh, 'w') as f:
        f.write(_cmds)
    gpu = wi%8
    if i == 0:
        target_tmux = "run-0"
        tmuxcmd = f"tmux new -s '{target_tmux}' -d 'CUDA_VISIBLE_DEVICES={gpu} sh {tmpsh} || echo Done sleep 300 && sleep 300'"
        
    else:
        tmuxcmd = f"tmux new-window -n w{wi} -t {target_tmux}: 'CUDA_VISIBLE_DEVICES={gpu} sh {tmpsh} || echo Done Sleep 300 && sleep 300'"
    wi += 1
    print(tmuxcmd)
    os.system(tmuxcmd)