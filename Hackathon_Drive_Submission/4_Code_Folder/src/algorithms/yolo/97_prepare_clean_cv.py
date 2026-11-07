"""Prepare ten YOLO folds from manually reviewed duplicate groups."""
from __future__ import annotations
import argparse,csv,random
from collections import Counter
from pathlib import Path
import yaml
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.git').is_dir())
CLASSES=['D00','D01','D10','D11','D20','D40','D43','D44','D50','D0w0']
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--images',type=Path,default=ROOT/'dataset/subsets/multicountry/RDD2022/yolo/images/train')
 ap.add_argument('--labels',type=Path,default=ROOT/'dataset/subsets/multicountry/RDD2022/yolo/labels/train')
 ap.add_argument('--reviewed-groups',type=Path,default=ROOT/'results/cross_validation_clean/reviewed_groups.csv')
 ap.add_argument('--output',type=Path,default=ROOT/'results/cross_validation_clean')
 ap.add_argument('--folds',type=int,default=10); ap.add_argument('--seed',type=int,default=42)
 a=ap.parse_args(); images=sorted(p for p in a.images.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
 if not images: raise SystemExit(f'No images found: {a.images}')
 if not a.reviewed_groups.is_file(): raise SystemExit(f'Missing reviewed groups CSV: {a.reviewed_groups}. Copy reviewed_groups_template.csv, fill group_id values, and save as reviewed_groups.csv.')
 with a.reviewed_groups.open(newline='',encoding='utf-8-sig') as f: reviewed=list(csv.DictReader(f))
 if not reviewed or not {'image','group_id'}.issubset(reviewed[0]): raise SystemExit('Reviewed groups CSV needs image and group_id columns.')
 group_by_stem={}
 for r in reviewed:
  stem=Path(r['image']).stem; group=str(r.get('group_id','')).strip()
  if not group: raise SystemExit(f'Blank group_id for {stem}; give single images their own unique IDs too.')
  if stem in group_by_stem and group_by_stem[stem]!=group: raise SystemExit(f'Conflicting groups for {stem}')
  group_by_stem[stem]=group
 rows=[]
 for im in images:
  label=a.labels/f'{im.stem}.txt'
  if not label.is_file(): raise SystemExit(f'Missing label: {label}')
  if im.stem not in group_by_stem: raise SystemExit(f'Image missing from reviewed groups CSV: {im.name}')
  present=set()
  for line in label.read_text(encoding='utf-8').splitlines():
   if not line.strip(): continue
   q=line.split(); cid=int(q[0])
   if len(q)!=5 or cid<0 or cid>=len(CLASSES): raise SystemExit(f'Invalid label line in {label}: {line}')
   present.add(CLASSES[cid])
  rows.append({'image':im.resolve(),'label':label.resolve(),'group':group_by_stem[im.stem],'classes':present,'country':im.stem.split('_')[0]})
 missing=set(group_by_stem)-{im.stem for im in images}
 if missing: raise SystemExit(f'Reviewed groups contain stems absent from active train images: {sorted(missing)[:5]}')
 by_group={}
 for i,r in enumerate(rows): by_group.setdefault(r['group'],[]).append(i)
 bundles=list(by_group.values()); rng=random.Random(a.seed); rng.shuffle(bundles)
 totals=Counter(c for r in rows for c in r['classes']); target_n=len(rows)/a.folds
 assigned=[[] for _ in range(a.folds)]; sizes=[0]*a.folds; counts=[Counter() for _ in range(a.folds)]
 bundles.sort(key=lambda g:(-sum(1/max(1,totals[c]) for i in g for c in rows[i]['classes']),-len(g)))
 for group in bundles:
  gc=Counter(c for i in group for c in rows[i]['classes'])
  def score(f):
   size_cost=sum(((sizes[k]+(len(group) if k==f else 0)-target_n)/target_n)**2 for k in range(a.folds))
   cls_cost=0.0
   for k in range(a.folds):
    for c in CLASSES:
     target=max(totals[c]/a.folds,0.25); value=counts[k][c]+(gc[c] if k==f else 0)
     cls_cost+=((value-target)/target)**2
   return size_cost+0.35*cls_cost
  fold=min(range(a.folds),key=score); assigned[fold].extend(group); sizes[fold]+=len(group); counts[fold].update(gc)
 if min(sizes)<0.85*target_n or any(counts[f]['D40']==0 for f in range(a.folds)):
  raise SystemExit(f'Unsafe fold balance: sizes={sizes}; D40 per fold={[c["D40"] for c in counts]}. Refine reviewed groups/stratification; no configs written.')
 output=a.output if a.output.is_absolute() else ROOT/a.output
 output.mkdir(parents=True,exist_ok=True)
 manifest=output/'grouped_fold_manifest.csv'
 with manifest.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['image','label','country','classes','group_id','fold']);w.writeheader()
  for fold,ids in enumerate(assigned,1):
   for i in ids:
    r=rows[i]; w.writerow({'image':r['image'].as_posix(),'label':r['label'].as_posix(),'country':r['country'],'classes':'|'.join(sorted(r['classes'])),'group_id':r['group'],'fold':fold})
 for fold,valid in enumerate(assigned,1):
  d=output/f'fold_{fold}';d.mkdir(parents=True,exist_ok=True); train=[i for f,ids in enumerate(assigned) if f!=fold-1 for i in ids]
  tr=d/'train.txt';va=d/'val.txt';tr.write_text('\n'.join(rows[i]['image'].as_posix() for i in train)+'\n',encoding='utf-8');va.write_text('\n'.join(rows[i]['image'].as_posix() for i in valid)+'\n',encoding='utf-8')
  cfg={'path':a.images.resolve().parent.parent.as_posix(),'train':tr.resolve().as_posix(),'val':va.resolve().as_posix(),'nc':len(CLASSES),'names':CLASSES}
  (d/'data.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
  print(f'Fold {fold}: train={len(train)} validation={len(valid)} images_with_D40={counts[fold-1]["D40"]}')
 print(f'Created manually grouped folds: {manifest}')
if __name__=='__main__':main()
