import argparse,json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer,AutoModel
from tqdm import tqdm
p=argparse.ArgumentParser()
p.add_argument("--model",choices=["contriever","medcpt"],required=True)
p.add_argument("--size",default="limit-small"); p.add_argument("--batch-size",type=int,default=8)
a=p.parse_args(); root=Path(__file__).resolve().parents[1]; data=root/"data"/a.size
out=root/"embeddings"/a.model/a.size; out.mkdir(parents=True,exist_ok=True)
if a.model=="contriever": qname=dname="facebook/contriever"
else: qname="ncbi/MedCPT-Query-Encoder"; dname="ncbi/MedCPT-Article-Encoder"
device="cuda" if torch.cuda.is_available() else "cpu"; print("Device:",device)
def read(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
q=read(data/"queries.jsonl"); d=read(data/"corpus.jsonl")
def pool(h,m):
    m=m.unsqueeze(-1).expand(h.size()).float()
    return (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
def enc(texts,name):
    tok=AutoTokenizer.from_pretrained(name); model=AutoModel.from_pretrained(name).to(device).eval(); out=[]
    with torch.no_grad():
        for i in tqdm(range(0,len(texts),a.batch_size),desc=name):
            e=tok(texts[i:i+a.batch_size],padding=True,truncation=True,max_length=512,return_tensors="pt")
            e={k:v.to(device) for k,v in e.items()}
            v=pool(model(**e).last_hidden_state,e["attention_mask"])
            out.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy())
    return np.concatenate(out)
qv=enc([x["text"] for x in q],qname); dv=enc([x["text"] for x in d],dname)
np.save(out/"queries.npy",qv.astype("float32")); np.save(out/"corpus.npy",dv.astype("float32"))
(out/"query_ids.txt").write_text("\n".join(x["_id"] for x in q),encoding="utf-8")
(out/"corpus_ids.txt").write_text("\n".join(x["_id"] for x in d),encoding="utf-8")
print("Query shape:",qv.shape); print("Corpus shape:",dv.shape)
