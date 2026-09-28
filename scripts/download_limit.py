import argparse, shutil, subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--size", choices=["limit-small","limit"], default="limit-small")
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
repo=root/"limit_repo"
if not repo.exists():
    subprocess.run(["git","clone","https://github.com/google-deepmind/limit.git",str(repo)],check=True)
src=repo/"data"/a.size
dst=root/"data"/a.size
dst.parent.mkdir(parents=True,exist_ok=True)
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src,dst)
print("LIMIT data copied to",dst)
