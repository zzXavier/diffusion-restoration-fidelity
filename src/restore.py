"""Invoke the pinned author's pipeline; collect the actual stage-1 tensor before diffusion."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
PIN='5c2d6c177c3f2965e6d33a32931248d242ef7b3d'

def locate():
    repo=Path(os.getenv('DIFFBIR_SOURCE',str(ROOT/'.cache'/'DiffBIR'))).resolve()
    repo.parent.mkdir(parents=True,exist_ok=True)
    if not repo.exists():
        subprocess.run(['git','clone','--filter=blob:none','https://github.com/XPixelGroup/DiffBIR.git',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'checkout','--detach',PIN],check=True)
    rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if rev!=PIN: raise ValueError(f'DiffBIR revision differs: {rev}')
    return repo

def restore(out):
    repo=locate()
    subprocess.run([sys.executable,str(Path(__file__).resolve()),str(out.resolve())],cwd=repo,check=True)

if __name__=='__main__':
    out=Path(sys.argv[1])
    sys.path.insert(0,str(Path.cwd()))
    import torch
    from PIL import Image
    import inference as upstream
    sys.argv=['inference.py','--task','sr','--version','v2.1','--upscale','2','--steps','10','--seed','29','--cfg_scale','4','--captioner','none','--precision','fp16','--cleaner_tiled','--vae_encoder_tiled','--vae_decoder_tiled','--cldm_tiled','--input',str(out/'degraded'),'--output',str(out/'diffusion')]
    args=upstream.parse_args()
    upstream.set_seed(args.seed)
    class Capture(upstream.BSRInferenceLoop):
        def load_pipeline(self):
            super().load_pipeline()
            origin=self.pipeline.apply_cleaner
            def save_stage1(*args,**kwargs):
                x=origin(*args,**kwargs)
                arr=(x[0].detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                p=out/'stage1';p.mkdir(exist_ok=True)
                Image.fromarray(arr).save(p/(self.loop_ctx['file_stem']+'.png'))
                return x
            self.pipeline.apply_cleaner=save_stage1
        def after_load_lq(self,lq):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            self.t0=time.perf_counter()
            return super().after_load_lq(lq)
        def save(self,samples,pos_prompt,neg_prompt):
            torch.cuda.synchronize()
            costs={'seconds':time.perf_counter()-self.t0,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20}
            (out/'diffusion'/(self.loop_ctx['file_stem']+'.json')).write_text(json.dumps(costs))
            super().save(samples,pos_prompt,neg_prompt)
    Capture(args).run()

