from __future__ import annotations
import csv, importlib.util, json
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())
DATA = ROOT / 'dataset/processed/cnn_lstm/test'
CHECKPOINT = ROOT / 'results/cnn_lstm/segmentation/best_cnn_lstm_spatial_v3.pt'
OUT = ROOT / 'results/cnn_lstm/segmentation/v3_independent_test_20261016'
DEVICE = torch.device('cpu')
THRESHOLD = 0.5

def main():
    if not CHECKPOINT.is_file(): raise SystemExit(f'Missing checkpoint: {CHECKPOINT}')
    if OUT.exists(): raise SystemExit(f'Refusing to overwrite: {OUT}')
    files = sorted(DATA.glob('*.npz'))
    if not files: raise SystemExit(f'No test NPZ files under {DATA}')
    script = ROOT / 'src/algorithms/cnn_lstm/58_train_cnn_lstm_segmentation.py'
    spec = importlib.util.spec_from_file_location('cnn_lstm_test_model', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.CNNLSTMSegmenter().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    OUT.mkdir(parents=True)
    rows=[]; total_inter=total_pred=total_true=total_correct=total_pixels=0
    with torch.no_grad():
        for path in files:
            with np.load(path) as data:
                frames=data['frames'].copy(); masks=data['masks'].copy()
            x=torch.from_numpy(frames).permute(0,3,1,2).float().unsqueeze(0)/255.0
            probs=torch.sigmoid(model(x))[0,:,0].cpu().numpy()
            target=masks>0; pred=probs>=THRESHOLD
            inter=int(np.logical_and(pred,target).sum()); pc=int(pred.sum()); tc=int(target.sum()); union=pc+tc-inter
            dice=2*inter/max(1,pc+tc); iou=inter/max(1,union); acc=float(np.mean(pred==target))
            total_inter+=inter;total_pred+=pc;total_true+=tc;total_correct+=int(np.sum(pred==target));total_pixels+=target.size
            rows.append({'video':path.name,'frames':int(len(frames)),'gt_positive_pixels':tc,'pred_positive_pixels':pc,'dice':dice,'iou':iou,'pixel_accuracy':acc,'gt_coverage_percent':100*float(target.mean()),'pred_coverage_percent':100*float(pred.mean())})
            frame=frames[0].copy(); gt=target[0]; pr=pred[0]; overlay=frame.copy()
            overlay[pr & gt]=[0,255,0];overlay[pr & ~gt]=[255,0,0];overlay[~pr & gt]=[0,0,255]
            panel=np.concatenate([frame, cv2.cvtColor(gt.astype('uint8')*255,cv2.COLOR_GRAY2RGB), cv2.cvtColor(pr.astype('uint8')*255,cv2.COLOR_GRAY2RGB), overlay],axis=1)
            cv2.imwrite(str(OUT/f'{path.stem}_preview.png'),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
            print(f'{path.name}: Dice={dice:.4f}, IoU={iou:.4f}, accuracy={acc:.4f}, pred coverage={100*float(pred.mean()):.2f}%',flush=True)
    summary={'model':'CNN + Spatial LSTM V3','checkpoint':str(CHECKPOINT.relative_to(ROOT)),'split':'independent processed test split','test_videos':len(files),'threshold':THRESHOLD,'global_dice':2*total_inter/max(1,total_pred+total_true),'global_iou':total_inter/max(1,total_pred+total_true-total_inter),'global_pixel_accuracy':total_correct/max(1,total_pixels),'mean_video_dice':float(np.mean([r['dice'] for r in rows])),'mean_video_iou':float(np.mean([r['iou'] for r in rows])),'per_video':rows,'note':'Segmentation test results apply only to the video mask task; not image damage-class detection.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'per_video.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='per_video'},indent=2));print('Output:',OUT)
if __name__=='__main__':main()
