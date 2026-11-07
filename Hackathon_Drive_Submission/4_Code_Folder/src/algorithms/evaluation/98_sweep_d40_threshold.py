"""Measure D40 detector precision/recall at confidence cutoffs on real validation."""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime
from pathlib import Path
import yaml
from ultralytics import YOLO
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.git').is_dir())
CHECKPOINT='results/training/yolo12n_30_epochs/weights/best.pt'
THRESHOLDS=[.01,.02,.03,.05,.075,.10,.15,.20,.25,.30,.40,.50]
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(1,aa+bb-inter)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=ROOT/'dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml');ap.add_argument('--checkpoint',default=CHECKPOINT);ap.add_argument('--imgsz',type=int,default=640);ap.add_argument('--output',type=Path);a=ap.parse_args()
 data_path=a.data if a.data.is_absolute() else ROOT/a.data; data=yaml.safe_load(data_path.read_text(encoding='utf-8')); names=data['names']; names=list(names.values()) if isinstance(names,dict) else names
 if 'D40' not in names: raise SystemExit('D40 missing from configured class names')
 cid=names.index('D40');base=Path(data['path']);base=base if base.is_absolute() else ROOT/base
 images_dir=base/data['val']; images_dir=images_dir if images_dir.is_absolute() else ROOT/images_dir
 labels_dir=base/'labels'/'val'; labels_dir=labels_dir if labels_dir.is_absolute() else ROOT/labels_dir
 images=sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
 if not images: raise SystemExit(f'No validation images under {images_dir}')
 model_path=ROOT/a.checkpoint
 if not model_path.is_file(): raise SystemExit(f'Checkpoint missing: {model_path}')
 model=YOLO(str(model_path)); ckpt_names=model.names; ckpt_names=list(ckpt_names.values()) if isinstance(ckpt_names,dict) else ckpt_names
 if ckpt_names!=names: raise SystemExit(f'Checkpoint and dataset class order differs: {ckpt_names} != {names}')
 gt={}; predictions={};
 for image in images:
  label_path=labels_dir/(image.stem+'.txt'); boxes=[]
  for line in label_path.read_text(encoding='utf-8').splitlines():
   q=line.split()
   if len(q)!=5 or int(q[0])!=cid:continue
   cx,cy,w,h=map(float,q[1:]); iwih=__import__('PIL').Image.open(image).size; iw,ih=iwih
   boxes.append(((cx-w/2)*iw,(cy-h/2)*ih,(cx+w/2)*iw,(cy+h/2)*ih))
  gt[str(image.resolve())]=boxes
  pred=next(model.predict(source=str(image),conf=.001,iou=.50,imgsz=a.imgsz,device='cpu',classes=[cid],max_det=300,verbose=False,stream=True))
  if pred.boxes is None: predictions[str(image.resolve())]=[];continue
  xy=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
  predictions[str(image.resolve())]=[(tuple(map(float,b)),float(s)) for b,s in zip(xy,scores)]
  print(f'{len(predictions):3d}/{len(images)} {image.name}',flush=True)
 rows=[]; total_gt=sum(map(len,gt.values()))
 for threshold in THRESHOLDS:
  tp=fp=0
  for key,boxes in predictions.items():
   matched=set()
   for box,score in sorted((x for x in boxes if x[1]>=threshold),key=lambda x:x[1],reverse=True):
    overlaps=[(iou(box,g),i) for i,g in enumerate(gt[key]) if i not in matched]
    best,index=max(overlaps,default=(0,-1))
    if best>=.5:tp+=1;matched.add(index)
    else:fp+=1
  fn=total_gt-tp;precision=tp/max(1,tp+fp);recall=tp/max(1,total_gt);f1=2*precision*recall/max(1e-12,precision+recall)
  rows.append({'confidence':threshold,'tp':tp,'fp':fp,'fn':fn,'ground_truth':total_gt,'precision_iou50':precision,'recall_iou50':recall,'f1_iou50':f1})
 output=a.output or ROOT/'results'/f'd40_threshold_sweep_{datetime.now():%Y%m%d_%H%M%S}';output=output if output.is_absolute() else ROOT/output
 output.mkdir(parents=True,exist_ok=False)
 with (output/'threshold_metrics.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 summary={'checkpoint':str(model_path.relative_to(ROOT)),'data':str(data_path.relative_to(ROOT)),'split':'real validation','images':len(images),'ground_truth_d40':total_gt,'iou_threshold':.5,'imgsz':a.imgsz,'thresholds':rows,'best_f1':max(rows,key=lambda r:r['f1_iou50']),'note':'Thresholds selected on validation are exploratory and require untouched test confirmation.'}
 (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));print(f'Output: {output}')
if __name__=='__main__':main()
