"""Evaluate the dashboard ViT hybrid D40 output on grouped real validation images."""
from __future__ import annotations
import argparse,csv,json,time,importlib.util
from datetime import datetime
from pathlib import Path
import cv2,yaml
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.git').is_dir())
APP=ROOT/'src/89_road_damage_app.py'
def load_app():
 s=importlib.util.spec_from_file_location('road_damage_app_eval',APP);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(1,aa+bb-inter)
def main():
 choices=tuple(name for name,cfg in load_app().MODELS.items() if cfg['kind'] in {'efficientnet','resnet','vit'})
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=ROOT/'dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml');ap.add_argument('--thresholds',default='0.03,0.10');ap.add_argument('--algorithm',choices=choices,default='YOLO12n + ViT-B/16');ap.add_argument('--detector-checkpoint',type=Path,help='Optional YOLO checkpoint to evaluate through the dashboard inference path.');ap.add_argument('--output',type=Path);a=ap.parse_args()
 dp=a.data if a.data.is_absolute() else ROOT/a.data;d=yaml.safe_load(dp.read_text(encoding='utf-8'));names=d['names']; names=list(names.values()) if isinstance(names,dict) else names; cid=names.index('D40')
 base=Path(d['path']);base=base if base.is_absolute() else ROOT/base; image_dir=base/d['val'];image_dir=image_dir if image_dir.is_absolute() else ROOT/image_dir;label_dir=base/'labels'/'val';label_dir=label_dir if label_dir.is_absolute() else ROOT/label_dir
 images=sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'});app=load_app();model_name=a.algorithm
 if a.detector_checkpoint:
  detector_checkpoint=a.detector_checkpoint if a.detector_checkpoint.is_absolute() else ROOT/a.detector_checkpoint
  if not detector_checkpoint.is_file():raise SystemExit(f'Missing detector checkpoint: {detector_checkpoint}')
  app.MODELS[model_name]['detector']=detector_checkpoint.relative_to(ROOT).as_posix()
 bundle=app.load_models(model_name)
 thresholds=[float(x) for x in a.thresholds.split(',')]; reports=[]; detailed=[]
 for threshold in thresholds:
  stats={'threshold':threshold,'images':len(images),'gt':0,'tp':0,'fp':0,'fn':0,'seconds':0.0,'counts':[]}
  for n,image in enumerate(images,1):
   frame=cv2.imread(str(image));h,w=frame.shape[:2];gt=[]
   for line in (label_dir/(image.stem+'.txt')).read_text(encoding='utf-8').splitlines():
    q=line.split()
    if len(q)!=5 or int(q[0])!=cid:continue
    cx,cy,bw,bh=map(float,q[1:]);gt.append(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h))
   started=time.perf_counter();_annotated,rows=app._infer_frame(frame,bundle,threshold,model_name,image.name);stats['seconds']+=time.perf_counter()-started
   preds=[]
   for row in rows:
    if row['detector_class']=='D40': preds.append(((row['x1'],row['y1'],row['x2'],row['y2']),row['detector_confidence'],row['classifier_class'],row['class_confidence']))
   matched=set();tp=0
   for box,score,label,clsconf in sorted(preds,key=lambda z:z[1],reverse=True):
    overlaps=[(iou(box,g),idx) for idx,g in enumerate(gt) if idx not in matched];best,idx=max(overlaps,default=(0,-1))
    is_match=best>=.5
    if is_match:matched.add(idx);tp+=1
    detailed.append({'threshold':threshold,'image':image.name,'detector_class':'D40','detector_confidence':score,'classifier_class':label,'classifier_confidence':clsconf,'xyxy':list(box),'best_iou':best,'matched':is_match})
   stats['gt']+=len(gt);stats['tp']+=tp;stats['fp']+=len(preds)-tp;stats['fn']+=len(gt)-tp;stats['counts'].append({'image':image.name,'gt':len(gt),'predictions':len(preds),'matched':tp})
   if n%25==0:print(f'threshold={threshold:.2f}: {n}/{len(images)}',flush=True)
  stats['precision']=stats['tp']/max(1,stats['tp']+stats['fp']);stats['recall']=stats['tp']/max(1,stats['gt']);stats['f1']=2*stats['precision']*stats['recall']/max(1e-12,stats['precision']+stats['recall']);reports.append(stats)
 out=a.output or ROOT/'results'/f'vit_hybrid_d40_eval_{datetime.now():%Y%m%d_%H%M%S}';out=out if out.is_absolute() else ROOT/out;out.mkdir(parents=True,exist_ok=False)
 summary={'pipeline':model_name,'checkpoint':app.MODELS[model_name],'data':str(dp.relative_to(ROOT)),'split':'real grouped validation','iou_threshold':.5,'threshold_reports':reports,'note':'Validation tuning only; not a test-set score.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:wri=csv.DictWriter(f,fieldnames=['threshold','image','detector_class','detector_confidence','classifier_class','classifier_confidence','xyxy','best_iou','matched']);wri.writeheader();wri.writerows(detailed)
 print(json.dumps([{k:v for k,v in r.items() if k not in ('counts',)} for r in reports],indent=2));print(f'Output: {out}')
if __name__=='__main__':main()
