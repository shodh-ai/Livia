import os

PATH_TO_CHECKPOINTS = "./checkpoints/prenorm-crms-3B/"
PATH_TO_DRIVE = "drive:GPT-NEOX/EXPERIMENT_WEIGHTS_SLIMPAJAMA_PRENORM_CRMS_3B/"


ckpt_paths = os.listdir(PATH_TO_CHECKPOINTS)
for i in ckpt_paths:
    cmd = f"rclone copy -v  --drive-chunk-size=8G --fast-list --transfers=16 --buffer-size=0 {os.path.join(PATH_TO_CHECKPOINTS,i)} {PATH_TO_DRIVE+i}"
    os.system(cmd)
    print(cmd)
    print("-"*30,"Done","-"*30)