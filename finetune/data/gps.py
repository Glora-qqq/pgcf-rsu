"""从 GPS 文本读取车辆出现的帧范围。"""

import os
import os
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

def find_gps_file(scene: str, vehicle: str, data_root: Path, is_test: bool = False) -> Optional[Path]:

    if data_root is None:
        raise ValueError("data_root参数不能为None")

    vehicle_dir = data_root / scene / vehicle

    if not vehicle_dir.exists():
        print(f"[WARNING] 车辆目录不存在: {vehicle_dir}")
        return None

    vehicle_num = vehicle.replace('Car', '').replace('car', '').replace('AT', '').replace('UAV', '')

    if is_test:
        possible_patterns = [
            f"{vehicle}_test_cat_gt.txt",
            f"{vehicle}_test_cat_noise.txt",
            f"{vehicle.lower()}_test_cat_gt.txt",
            f"{vehicle.lower()}_test_cat_noise.txt",
            f"car{vehicle_num}_test_cat_gt.txt",
            f"car{vehicle_num}_test_cat_noise.txt",
            f"Car{vehicle_num}_test_cat_gt.txt",
            f"Car{vehicle_num}_test_cat_noise.txt",
        ]
    else:
        possible_patterns = [
            f"{vehicle}_train_cat_gt.txt",
            f"{vehicle.lower()}_train_cat_gt.txt",
            f"car{vehicle_num}_train_cat_gt.txt",
            f"{vehicle}_train_cat_gt_noise.txt",
            f"{vehicle.lower()}_train_cat_gt_noise.txt",
            f"car{vehicle_num}_train_cat_gt_noise.txt",
        ]

    for pattern in possible_patterns:
        gps_file = vehicle_dir / pattern
        if gps_file.exists():
            return gps_file

    print(f"[WARNING] 未找到GPS文件: {vehicle_dir}/[{', '.join(possible_patterns[:3])}...]")
    return None

def extract_frame_range_from_gps(gps_file_path: Path) -> Tuple[int, int, int]:

    try:
        data = np.loadtxt(gps_file_path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        frame_ids = data[:, 6].astype(int)

        start_frame = int(frame_ids[0])
        end_frame = int(frame_ids[-1])
        total_frames = len(frame_ids)

        expected_frames = end_frame - start_frame + 1
        if total_frames != expected_frames:
            print(f"[WARNING] GPS帧不连续: 预期{expected_frames}帧，实际{total_frames}帧")

        return start_frame, end_frame, total_frames

    except Exception as e:
        error_msg = f"[ERROR] 读取GPS文件失败: {gps_file_path}\n错误: {e}"
        raise RuntimeError(error_msg)

def extract_gps_coordinates(gps_file_path: Path) -> np.ndarray:

    try:
        data = np.loadtxt(gps_file_path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        coordinates = data[:, :3]

        return coordinates

    except Exception as e:
        error_msg = f"[ERROR] 读取GPS坐标失败: {gps_file_path}\n错误: {e}"
        raise RuntimeError(error_msg)

def get_vehicle_frame_info(scene: str, vehicle: str, data_root: Path, is_test: bool = False) -> Dict:

    gps_file = find_gps_file(scene, vehicle, data_root, is_test)

    if gps_file is None:
        data_type = "测试集" if is_test else "训练集"
        error_msg = f"[ERROR] 未找到GPS文件: {data_type} {scene}/{vehicle}\n" \
                   f"   检查路径: {data_root}/{scene}/{vehicle}/"
        raise FileNotFoundError(error_msg)

    start_frame, end_frame, total_frames = extract_frame_range_from_gps(gps_file)

    start_index = start_frame - 1
    end_index = end_frame - 1

    expected_length = end_index - start_index + 1
    if expected_length != total_frames:
        print(f"[WARNING] 索引计算不一致: {expected_length} vs {total_frames}")

    gps_coordinates = extract_gps_coordinates(gps_file)

    frame_ids = np.arange(start_frame, end_frame + 1)

    return {
        'gps_file': str(gps_file),
        'start_frame': start_frame,
        'end_frame': end_frame,
        'total_frames': total_frames,
        'start_index': start_index,
        'end_index': end_index,
        'gps_coordinates': gps_coordinates,
        'frame_ids': frame_ids,
    }

def get_scene_vehicles_frame_info(
    scene: str,
    vehicles: List[str],
    data_root: Path,
    is_test: bool = False
) -> Dict[str, Dict]:

    vehicles_info = {}

    for vehicle in vehicles:
        try:
            frame_info = get_vehicle_frame_info(scene, vehicle, data_root, is_test)
            vehicles_info[vehicle] = frame_info
        except FileNotFoundError as e:
            print(f"[WARNING] 跳过车辆 {vehicle}: {e}")
            continue

    return vehicles_info

if __name__ == '__main__':
    print("=" * 70)
    print("GPS加载器测试")
    print("=" * 70)

    print("\n【测试1：训练集单车辆】")
    train_data_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')) / 'train'
    scene = 'Scene1'
    vehicle = 'Car1'

    try:
        info = get_vehicle_frame_info(scene, vehicle, train_data_root, is_test=False)
        print(f"✓ {scene}/{vehicle}")
        print(f"  GPS文件: {Path(info['gps_file']).name}")
        print(f"  帧范围: {info['start_frame']}-{info['end_frame']} (1-based)")
        print(f"  Python索引: [{info['start_index']}:{info['end_index']+1}] (0-based)")
        print(f"  总帧数: {info['total_frames']}")
        print(f"  GPS坐标: {info['gps_coordinates'].shape}")
        print(f"  验证: {info['end_index'] - info['start_index'] + 1} == {info['total_frames']} ✓")
    except Exception as e:
        print(f"✗ 错误: {e}")

    print("\n【测试2：测试集单车辆】")
    test_data_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')) / 'test'
    scene = 'Scene_1'
    vehicle = 'Car1'

    try:
        info = get_vehicle_frame_info(scene, vehicle, test_data_root, is_test=True)
        print(f"✓ {scene}/{vehicle}")
        print(f"  GPS文件: {Path(info['gps_file']).name}")
        print(f"  帧范围: {info['start_frame']}-{info['end_frame']} (1-based)")
        print(f"  Python索引: [{info['start_index']}:{info['end_index']+1}] (0-based)")
        print(f"  总帧数: {info['total_frames']}")
        print(f"  验证: {info['end_index'] - info['start_index'] + 1} == {info['total_frames']} ✓")
    except Exception as e:
        print(f"✗ 错误: {e}")

    print("\n【测试3：训练集批量车辆】")
    vehicles = ['Car1', 'Car4', 'Car5']

    try:
        vehicles_info = get_scene_vehicles_frame_info(
            scene='Scene1',
            vehicles=vehicles,
            data_root=train_data_root,
            is_test=False
        )

        print(f"✓ 成功加载 {len(vehicles_info)}/{len(vehicles)} 个车辆")
        for veh, info in vehicles_info.items():
            print(f"  {veh}: 帧{info['start_frame']}-{info['end_frame']} ({info['total_frames']}帧)")
    except Exception as e:
        print(f"✗ 错误: {e}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
