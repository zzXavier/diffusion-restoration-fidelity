"""Run only the real DiffBIR v2.1 SwinIR cleaner without loading the diffusion model."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys
import time
import json
from PIL import Image
import numpy as np
import requests
import torch
import yaml
if __package__:
    from .restore import locate
else:
    from restore import locate

FILENAME='realesrgan_s4_swinir_100k.pth'
SHA='577b8f84815d324e5ee17ef3c85d8bc9b7ebd05b20a6d61a5452b66e7e24dcaf'
REV='fd6263503564bfbfa156665a90832e1b729434f4'

def fetch(dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==SHA: return dest
    parts=[]
    from concurrent.futures import ThreadPoolExecutor
    total=90208287;chunk=4*1024*1024
    def part(i):
        start=i*chunk;end=min(start+chunk,total)-1
        partpath=dest.with_name(FILENAME+'.'+str(i))
        if not partpath.exists() or partpath.stat().st_size!=end-start+1:
            response=requests.get(f'https://huggingface.co/lxq007/DiffBIR-v2/resolve/{REV}/{FILENAME}',params={'part':i},headers={'Range':f'bytes={start}-{end}'},timeout=90)
            response.raise_for_status()
            if len(response.content)!=end-start+1: raise ValueError('Truncated weight fragment')
            partpath.write_bytes(response.content)
        return partpath
    with ThreadPoolExecutor(max_workers=5) as pool:
        parts=list(pool.map(part,range((total+chunk-1)//chunk)))
    tmp=dest.with_suffix('.assembling')
    with tmp.open('wb') as f:
        for p in parts:f.write(p.read_bytes())
    with tmp.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=SHA:raise ValueError('SwinIR SHA mismatch')
    tmp.replace(dest)
    for p in parts:p.unlink()
    return dest

def predict(out,device):
    torch.set_num_threads(4)
    root=locate()
    weights=fetch(root.parent/FILENAME)
    config=yaml.safe_load((root/'configs'/'inference'/'swinir.yaml').read_text())['params']
    spec=importlib.util.spec_from_file_location('diffbir_swinir',root/'diffbir'/'model'/'swinir.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.SwinIR(**config).eval()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.to(device)
    dest=out/'stage1';dest.mkdir(exist_ok=True)
    records={}
    for path in sorted((out/'degraded').glob('*.png')):
        source=Image.open(path).convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
        x=torch.from_numpy(np.array(source)).permute(2,0,1).unsqueeze(0).to(device).float()/255
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        t0=time.perf_counter()
        with torch.inference_mode():
            y=model(x)[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()
        records[path.name]={'wall_seconds':time.perf_counter()-t0,
            'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None}
        Image.fromarray((y*255).round().astype('uint8')).save(dest/path.name)
    (dest/'runtime.json').write_text(json.dumps({'device':device,'torch':torch.__version__,'weights':SHA,'cases':records},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();predict(args.output,args.device)

