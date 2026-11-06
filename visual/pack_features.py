"""把融合后的 npy 打包成微调阶段读取的 npz。"""

import argparse
from pathlib import Path

import numpy as np

def pack_one(scene_dir: Path, out_dir: Path) -> int:
    written = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for npy in sorted(scene_dir.rglob('*_fused_env_feature.npy')):
        stem = npy.name.replace('_fused_env_feature.npy', '')
        payload = {'fused': np.load(npy)}
        for view in ('left', 'middle', 'right'):
            view_path = npy.with_name(f'{stem}_{view}_env_feature.npy')
            if view_path.exists():
                payload[view] = np.load(view_path)
        out_file = out_dir / f'{stem}_features.npz'
        np.savez(out_file, **payload)
        written += 1
    return written

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fused-dir', type=Path, required=True, help='infer_fusion.py 的输出目录')
    parser.add_argument('--out-dir', type=Path, required=True, help='例如 finetune/fusion-train-RGB')
    args = parser.parse_args()
    count = pack_one(args.fused_dir, args.out_dir)
    print(f'wrote {count} npz files to {args.out_dir}')

if __name__ == '__main__':
    main()
