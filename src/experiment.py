"""Colab-sized experiments; faithful architecture, deliberately reduced protocol."""
import argparse, copy, csv, json, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .models import PowerMLP, MLP, SplineKAN, count_parameters


def make_data(seed, n=2048):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n,2,generator=generator)*2-1
    # Paper section 5.4 ablation target. Domain/split are our choices.
    y = x[:,0:1]*torch.exp(-x[:,1:2])
    return x,y


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def benchmark(model,x,steps=100):
    # Separate steady-state forward/backward microbenchmark; includes no optimizer.
    for _ in range(10):
        model.zero_grad(set_to_none=True); model(x).square().mean().backward()
    synchronize(x.device); start=time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True); model(x).square().mean().backward()
    synchronize(x.device)
    return (time.perf_counter()-start)*1000/steps


def run(steps=1000,seeds=(0,1,2),size='large',device='cpu',out='results'):
    if steps < 1 or not seeds: raise ValueError('positive steps and at least one seed required')
    torch.set_num_threads(1)
    device=torch.device(device)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    power_shape=(2,4,1) if size=='small' else (2,32,8,1)
    mlp_shape=(2,6,1) if size=='small' else (2,32,18,1)
    kan_shape=(2,1,1) if size=='small' else (2,2,1,1)
    grid=3 if size=='small' else 100
    factories={
      'PowerMLP':lambda:PowerMLP(power_shape),
      'MLP':lambda:MLP(mlp_shape),
      'SplineKAN':lambda:SplineKAN(kan_shape,grid),
      'No basis':lambda:PowerMLP(power_shape,basis=False),
      'ReLU instead of ReLU^3':lambda:PowerMLP(power_shape,k=1),
    }
    rows=[]; histories={}; predictions={}
    for seed in seeds:
      train_x,train_y=make_data(1000+seed,1024)
      val_x,val_y=make_data(2000+seed,256)
      test_x,test_y=make_data(3000+seed,1024)
      train_x,train_y,val_x,val_y,test_x,test_y=[t.to(device) for t in (train_x,train_y,val_x,val_y,test_x,test_y)]
      for name,factory in factories.items():
        torch.manual_seed(seed)
        model=factory().to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        best=float('inf');state=None;history=[];best_step=0
        synchronize(device);start=time.perf_counter()
        for step in range(1,steps+1):
          model.train();opt.zero_grad(set_to_none=True)
          loss=nn.functional.mse_loss(model(train_x),train_y)
          if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {name}, seed={seed}')
          loss.backward();opt.step()
          if step==1 or step%25==0 or step==steps:
            model.eval()
            with torch.no_grad(): val=nn.functional.mse_loss(model(val_x),val_y).item()
            history.append({'step':step,'train_mse':loss.item(),'val_rmse':val**0.5})
            if val < best: best=val;state=copy.deepcopy(model.state_dict());best_step=step
        synchronize(device);elapsed=time.perf_counter()-start
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
          pred=model(test_x);error=pred-test_y
          rmse=error.square().mean().sqrt().item();mae=error.abs().mean().item()
          r2=1-error.square().sum().item()/((test_y-test_y.mean()).square().sum().item())
        ms=benchmark(model,train_x[:256])
        row=dict(model=name,seed=seed,parameters=count_parameters(model),test_rmse=rmse,test_mae=mae,test_r2=r2,training_seconds=elapsed,step_ms=ms,best_step=best_step)
        rows.append(row);histories[f'{name}/{seed}']=history
        if seed==seeds[0]:
          predictions[name]=pred.cpu().numpy().ravel().tolist()
          torch.save(model.state_dict(),out/(name.replace(' ','_').replace('^','')+'.pt'))
        print(json.dumps(row),flush=True)
    summary=[]
    for name in factories:
      subset=[r for r in rows if r['model']==name]
      item={'model':name,'parameters':subset[0]['parameters']}
      for metric in ('test_rmse','test_mae','test_r2','training_seconds','step_ms'):
        values=[r[metric] for r in subset]
        item[metric+'_mean']=float(np.mean(values))
        item[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
      summary.append(item)
    results={'scope':'Reduced protocol, exact PowerMLP architecture; custom fixed-grid KAN, not pykan runtime or full paper reproduction',
      'config':{'steps':steps,'seeds':list(seeds),'size':size,'device':str(device),'domain':[-1,1],'train':1024,'validation':256,'test':1024,'optimizer':'Adam','learning_rate':.001,'batch':'full','selection':'lowest validation MSE, checked every 25 steps','target':'x * exp(-y)'},
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),'accelerator':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'},
      'runs':rows,'summary':summary}
    (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'history.json').write_text(json.dumps(histories,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    plot_results(out,histories,summary)
    return results


def plot_results(out,histories,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5))
    for key,series in histories.items():
      if key.endswith('/0'):
        ax.semilogy([r['step'] for r in series],[r['val_rmse'] for r in series],label=key.rsplit('/',1)[0])
    ax.set(xlabel='Training step',ylabel='Validation RMSE',title='Convergence on x exp(-y), seed 0');ax.legend();fig.tight_layout()
    fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar([r['model'] for r in summary],[r['test_rmse_mean'] for r in summary],yerr=[r['test_rmse_std'] for r in summary],capsize=4,color='#167d9a')
    ax.set(ylabel='Test RMSE (mean ± sample SD)',title='Independent test split; lower is better')
    ax.tick_params(axis='x',labelsize=9);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--size',choices=['small','large'],default='large');p.add_argument('--device',default='cpu');p.add_argument('--out',default='results')
    run(**vars(p.parse_args()))
