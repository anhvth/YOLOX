    """_summary_
    Example: python scripts/run.py ./scripts/video_extract_face_food.sh "/data/DMS_Behavior_Detection/mobile_cigarret_foreignerUS/testing/*/*/*.mp4" 8   
    """
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
    cmd = f"{args.shfile} {path}"
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
        tmuxcmd = f"tmux new -s '{target_tmux}' -d 'CUDA_VISIBLE_DEVICES={gpu} sh {tmpsh} || echo Done && sleep 10'"
    else:
        tmuxcmd = f"tmux new-window -n w{wi} -t {target_tmux}: 'CUDA_VISIBLE_DEVICES={gpu} sh {tmpsh} || echo Done && sleep 10'"
    wi += 1
    print(tmuxcmd)
    os.system(tmuxcmd)