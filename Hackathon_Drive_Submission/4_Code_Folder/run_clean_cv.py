"""Fail-closed runner for leakage-controlled YOLO cross-validation."""
from __future__ import annotations
import argparse, csv, datetime, json, statistics
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
CLASSES=['D00','D01','D10','D11','D20','D40','D43','D44','D50','D0w0']

def preflight(weights:Path, folds:int, group_csv:Path):
    if not weights.is_file(): raise SystemExit(f'Generic pretrained weights not found: {weights}')
    try: weights.relative_to(ROOT/'results')
    except ValueError: pass
    else: raise SystemExit('Refusing project-trained checkpoint: choose generic pretrained weights outside results/.')
    if group_csv.is_file():
        with group_csv.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        if not rows or not {'image','group_id'}.issubset(rows[0]): raise SystemExit('Reviewed groups CSV requires image,group_id columns.')
        grouped={Path(r['image']).stem:r['group_id'] for r in rows}
    else: raise SystemExit(f'Missing manually reviewed duplicate groups file: {group_csv}')
    configs=[]; fold_image_sets=[]
    for i in range(1,folds+1):
        p=ROOT/'results/cross_validation_clean'/f'fold_{i}'/'data.yaml'
        if not p.is_file(): raise SystemExit(f'Missing clean fold config: {p}. Review image groups and prepare folds first.')
        cfg=yaml.safe_load(p.read_text(encoding='utf-8')); names=cfg.get('names')
        if isinstance(names,dict): names=[names[k] for k in sorted(names)]
        if cfg.get('nc')!=len(CLASSES) or names!=CLASSES: raise SystemExit(f'Fold {i} class map mismatch: {names}')
        def read_list(key):
            listpath=Path(cfg[key]); listpath=listpath if listpath.is_absolute() else p.parent/listpath
            base=Path(cfg.get('path',p.parent)); base=base if base.is_absolute() else p.parent/base
            if listpath.suffix.lower()=='.txt': vals=[Path(line.strip()) for line in listpath.read_text(encoding='utf-8').splitlines() if line.strip()]
            else: raise SystemExit(f'Fold {i} {key} must point to a text image list.')
            return [v if v.is_absolute() else base/v for v in vals]
        train,val=read_list('train'),read_list('val')
        tr={str(x.resolve()) for x in train}; va={str(x.resolve()) for x in val}
        if not tr or not va or tr&va: raise SystemExit(f'Fold {i} has empty or overlapping train/validation images.')
        for q in train+val:
            if not q.is_file(): raise SystemExit(f'Fold {i} image is missing: {q}')
            if q.stem not in grouped: raise SystemExit(f'Image absent from reviewed groups CSV: {q}')
            parts=list(q.parts)
            if 'images' not in parts: raise SystemExit(f'Cannot derive YOLO label path from image: {q}')
            pos=parts.index('images')
            label=Path(*parts[:pos],'labels',*parts[pos+1:-1],q.stem+'.txt')
            if not label.is_file(): raise SystemExit(f'Fold {i} label is missing: {label}')
        trgroups={grouped[Path(x).stem] for x in tr}; vagroups={grouped[Path(x).stem] for x in va}
        if trgroups & vagroups: raise SystemExit(f'Fold {i} splits reviewed near-duplicate groups across train and validation.')
        fold_image_sets.append((tr,va))
        pothole_images=0
        for q in val:
            parts=list(q.parts); pos=parts.index('images'); label=Path(*parts[:pos],'labels',*parts[pos+1:-1],q.stem+'.txt')
            if any(line.split() and line.split()[0]=='5' for line in label.read_text(encoding='utf-8').splitlines()): pothole_images+=1
        if pothole_images==0: raise SystemExit(f'Fold {i} validation has no D40 examples.')
        configs.append((i,p,len(tr),len(va)))
    all_val=[image for _tr,va in fold_image_sets for image in va]
    universe=set(all_val)
    if len(all_val)!=len(universe): raise SystemExit('Validation folds overlap or repeat images.')
    for i,(tr,va) in enumerate(fold_image_sets,1):
        if tr != universe-va: raise SystemExit(f'Fold {i} training set is not the complement of the shared CV image set.')
    return configs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True,help='Local generic pretrained YOLO checkpoint outside results/.')
    ap.add_argument('--reviewed-groups',type=Path,default=ROOT/'results/cross_validation_clean/reviewed_groups.csv')
    ap.add_argument('--folds',type=int,default=10); ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--imgsz',type=int,default=640); ap.add_argument('--batch',type=int,default=2); ap.add_argument('--device',default='cpu')
    ap.add_argument('--train',action='store_true',help='Actually train. Without this flag, only preflight is performed.')
    a=ap.parse_args(); weights=a.weights if a.weights.is_absolute() else ROOT/a.weights
    groups=a.reviewed_groups if a.reviewed_groups.is_absolute() else ROOT/a.reviewed_groups
    configs=preflight(weights.resolve(),a.folds,groups.resolve())
    print(json.dumps({'preflight':'passed','weights':str(weights.resolve()),'folds':[{'fold':i,'train_images':tr,'validation_images':va} for i,_,tr,va in configs],'epochs':a.epochs,'imgsz':a.imgsz,'batch':a.batch,'device':a.device,'training_requested':a.train},indent=2))
    if not a.train: return
    from ultralytics import YOLO, settings
    out=ROOT/'results/cross_validation_clean/training'/datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
    if out.exists(): raise SystemExit(f'Refusing to overwrite output: {out}')
    settings.update({'runs_dir':str((ROOT/'results').resolve())})
    records=[]; status_file=out/'status.json'
    state={'state':'running','weights':str(weights.resolve()),'epochs':a.epochs,'folds_requested':len(configs),'completed_folds':[]}
    status_file.write_text(json.dumps(state,indent=2),encoding='utf-8')
    for i,cfg,_,_ in configs:
        state['current_fold']=i; status_file.write_text(json.dumps(state,indent=2),encoding='utf-8')
        try:
            model=YOLO(str(weights.resolve()))
            result=model.train(data=str(cfg.resolve()),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,workers=0,project=str(out.resolve()),name=f'fold_{i}',exist_ok=False,pretrained=True,plots=False,save=True,val=True,amp=False,cache=False,seed=42+i,deterministic=True,patience=a.epochs)
        except KeyboardInterrupt:
            state['state']='interrupted'; status_file.write_text(json.dumps(state,indent=2),encoding='utf-8'); raise
        metrics={k:float(v) for k,v in getattr(result,'results_dict',{}).items() if isinstance(v,(int,float))}
        records.append({'fold':i,'metrics':metrics,'best_checkpoint':str((out/f'fold_{i}'/'weights'/'best.pt'))})
        (out/f'fold_{i}'/'fold_metrics.json').write_text(json.dumps(records[-1],indent=2),encoding='utf-8')
        state['completed_folds'].append(i); status_file.write_text(json.dumps(state,indent=2),encoding='utf-8')
    summary={'weights':str(weights.resolve()),'epochs':a.epochs,'folds':records,'means':{},'stddevs':{}}
    keys=set.intersection(*(set(r['metrics']) for r in records)) if records else set()
    for k in keys:
        vals=[r['metrics'][k] for r in records]; summary['means'][k]=statistics.mean(vals); summary['stddevs'][k]=statistics.stdev(vals) if len(vals)>1 else 0.0
    (out/'cross_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    state['state']='completed'; state.pop('current_fold',None); status_file.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(f'Completed clean CV outputs: {out}')
if __name__=='__main__':main()
