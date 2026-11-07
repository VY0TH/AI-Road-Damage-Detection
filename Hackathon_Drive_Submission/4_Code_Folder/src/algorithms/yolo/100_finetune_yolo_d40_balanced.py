"""Fine-tune a separate YOLO checkpoint with real D40 train-image oversampling."""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path
import torch,yaml
from ultralytics import YOLO
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.git').is_dir())
DATA=ROOT/'dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml'
BASE=ROOT/'results/training/yolo12n_30_epochs/weights/best.pt'
RUN=ROOT/'results/training/yolo12n_d40_balanced_10ep_20261015'
PREP=ROOT/'results/d40_balanced_training_data_20261015'

def main():
 if not torch.cuda.is_available(): device='cpu'
 else: device='0'
 if RUN.exists(): raise SystemExit(f'Refusing to overwrite existing output: {RUN}')
 d=yaml.safe_load(DATA.read_text(encoding='utf-8')); names=d['names']; names=list(names.values()) if isinstance(names,dict) else names
 if names[5]!='D40': raise SystemExit(f'Expected D40 at class ID 5, got {names[5]}')
 base=Path(d['path']);base=base if base.is_absolute() else ROOT/base
 train_dir=base/'images'/'train'; label_dir=base/'labels'/'train'; val_dir=base/'images'/'val'; val_label_dir=base/'labels'/'val'
 images=sorted(p for p in train_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
 if not images: raise SystemExit(f'No training images: {train_dir}')
 d40=[]; classes=Counter()
 for image in images:
  label=label_dir/(image.stem+'.txt')
  if not label.is_file(): raise SystemExit(f'Missing train annotation: {label}')
  found=False
  for line in label.read_text(encoding='utf-8').splitlines():
   q=line.split()
   if len(q)!=5: raise SystemExit(f'Malformed label in {label}: {line}')
   cls=int(q[0]); classes[cls]+=1; found |= cls==5
  if found:d40.append(image)
 val_images=sorted(p for p in val_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
 if any(not (val_label_dir/(p.stem+'.txt')).is_file() for p in val_images): raise SystemExit('A validation label file is missing.')
 if not d40: raise SystemExit('No D40 training images found.')
 # Two extra occurrences per real D40 image: no synthetic samples and no validation data.
 PREP.mkdir(parents=True,exist_ok=False)
 train_list=PREP/'train_d40_oversampled.txt'; lines=[p.resolve().as_posix() for p in images]
 lines += [p.resolve().as_posix() for p in d40 for _ in range(2)]
 train_list.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 cfg={'path':base.as_posix(),'train':train_list.resolve().as_posix(),'val':d['val'],'nc':len(names),'names':names}
 custom_yaml=PREP/'data.yaml';custom_yaml.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
 manifest={'base_checkpoint':str(BASE.relative_to(ROOT)),'data_yaml':str(custom_yaml.relative_to(ROOT)),'real_train_images':len(images),'real_D40_train_images':len(d40),'train_samples_after_resampling':len(lines),'extra_D40_occurrences':2*len(d40),'validation_images_real_only':len(val_images),'epochs':10,'imgsz':640,'batch':2,'device':device,'class_counts_objects':dict(classes)}
 (PREP/'training_plan.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
 model=YOLO(str(BASE))
 model.train(data=str(custom_yaml),epochs=10,imgsz=640,batch=2,device=device,optimizer='AdamW',lr0=0.0005,lrf=0.05,cos_lr=True,weight_decay=0.0005,patience=10,seed=2026,deterministic=True,project=str(RUN.parent),name=RUN.name,exist_ok=False,save=True,plots=True,val=True,amp=False,cache=False,close_mosaic=5,mosaic=0.5,mixup=0.0,copy_paste=0.0,fliplr=0.5,flipud=0.0)
 best=RUN/'weights'/'best.pt'
 if not best.is_file(): raise SystemExit('Training ended without best.pt; do not select this run.')
 print(f'Candidate checkpoint: {best}')
 print(f'Training results: {RUN}')
if __name__=='__main__':main()

