import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--model",choices=["contriever","medcpt"],required=True); p.add_argument("--size",default="limit-small")
a=p.parse_args(); root=Path(__file__).resolve().parents[1]
for s in ["encode_dense.py","make_dense_run.py"]:
    cmd=[sys.executable,str(root/"scripts"/s),"--model",a.model,"--size",a.size]
    subprocess.run(cmd,check=True)
subprocess.run([sys.executable,str(root/"scripts"/"evaluate.py"),"--size",a.size,
                "--run",str(root/"results"/f"{a.model}_{a.size}.txt")],check=True)
