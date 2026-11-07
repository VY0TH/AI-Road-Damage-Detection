from __future__ import annotations
import csv, importlib.util, json, time
from pathlib import Path
import cv2, yaml
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.git').is_dir())
APP=ROOT/'src/89_road_damage_app.py';DATA=ROOT/'dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml';OUT=ROOT/'results/large_box_filter_eval_20261016'
CONF=.075;IOU=.50;AREA_CANDIDATE=.12;CLS_CONF_CANDIDATE=.90

def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(1,aa+bb-inter)
def metrics(gt,pred,names):
 out={};tot={'tp':0,'fp':0,'fn':0}
 for cls in names:
  tp=fp=0
  for key,boxes in gt.items():
   gs=[b for c,b in boxes if c==cls];ps=[p for p in pred[key] if p['class']==cls];matched=set()
   for p in sorted(ps,key=lambda x:x['score'],reverse=True):
    overlaps=[(iou(p['box'],g),i) for i,g in enumerate(gs) if i not in matched];v,k=max(overlaps,default=(0,-1))
    if v>=IOU:tp+=1;matched.add(k)
    else:fp+=1
  fn=sum(sum(1 for c,_ in boxes if c==cls) for boxes in gt.values())-tp
  if tp+fp+fn: out[cls]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}
  tot['tp']+=tp;tot['fp']+=fp;tot['fn']+=fn
 tot.update(precision=tot['tp']/max(1,tot['tp']+tot['fp']),recall=tot['tp']/max(1,tot['tp']+tot['fn']),f1=2*tot['tp']/max(1,2*tot['tp']+tot['fp']+tot['fn']))
 return {'per_class':out,'micro':tot}
def main():
 d=yaml.safe_load(DATA.read_text());names=d['names'];names=list(names.values()) if isinstance(names,dict) else names;base=Path(d['path']);base=base if base.is_absolute() else ROOT/base;idir=base/d['val'];idir=idir if idir.is_absolute() else ROOT/idir;ldir=base/'labels/val'
 spec=importlib.util.spec_from_file_location('app_largebox_eval',APP);app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app);model='YOLO12n + ViT-B/16';bundle=app.load_models(model)
 gt={};raw={};started=time.perf_counter();images=sorted(p for p in idir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
 for n,img in enumerate(images,1):
  frame=cv2.imread(str(img));h,w=frame.shape[:2];g=[]
  for line in (ldir/f'{img.stem}.txt').read_text().splitlines():
   q=line.split()
   if len(q)!=5:continue
   cls=names[int(q[0])];cx,cy,bw,bh=map(float,q[1:]);g.append((cls,((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h)))
  _,rows=app._infer_frame(frame,bundle,CONF,model,img.name);pred=[]
  for r in rows:
   area=max(1,(r['x2']-r['x1'])*(r['y2']-r['y1']))/(w*h)
   pred.append({'class':r['detector_class'],'score':r['detector_confidence'],'classifier_class':r['classifier_class'],'classifier_confidence':r['class_confidence'],'area_ratio':area,'box':(r['x1'],r['y1'],r['x2'],r['y2'])})
  gt[img.name]=g;raw[img.name]=pred
  if n%25==0:print(f'{n}/{len(images)}',flush=True)
 strict={k:[p for p in ps if p['class']=='D40' or not (p['area_ratio']>AREA_CANDIDATE and (p['classifier_class']!=p['class'] or p['classifier_confidence']<CLS_CONF_CANDIDATE))] for k,ps in raw.items()}
 summary={'pipeline':model,'checkpoint':app.MODELS[model]['detector'],'classifier_checkpoint':app.MODELS[model]['classifier'],'data':'grouped real validation','images':len(images),'ground_truth':sum(len(v) for v in gt.values()),'iou_threshold':IOU,'dashboard_current_filter':metrics(gt,raw,names),'candidate_broad_box_filter':{'area_threshold':AREA_CANDIDATE,'required_classifier_confidence':CLS_CONF_CANDIDATE,'metrics':metrics(gt,strict,names)},'runtime_seconds':time.perf_counter()-started,'note':'Candidate rule applies only to non-D40 proposals; the dashboard UI exposes detector and crop-classifier labels separately. Validation-only ablation.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
  fields=['image','class','score','classifier_class','classifier_confidence','area_ratio','x1','y1','x2','y2','gt_class','best_iou']
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
  for key,ps in raw.items():
   for p in ps:
    same=[box for cls,box in gt[key] if cls==p['class']];v=max((iou(p['box'],b) for b in same),default=0)
    wr.writerow({'image':key,'class':p['class'],'score':p['score'],'classifier_class':p['classifier_class'],'classifier_confidence':p['classifier_confidence'],'area_ratio':p['area_ratio'],'x1':p['box'][0],'y1':p['box'][1],'x2':p['box'][2],'y2':p['box'][3],'gt_class':p['class'] if v>=IOU else 'none','best_iou':v})
 print(json.dumps({'current':summary['dashboard_current_filter'],'candidate':summary['candidate_broad_box_filter']},indent=2));print('Output:',OUT)
if __name__=='__main__':main()
