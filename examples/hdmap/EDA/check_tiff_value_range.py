#!/usr/bin/env python3
"""HDMAP TIFF 파일 값 범위 검증 스크립트.

1000_tiff_minmax 폴더의 TIFF 파일들이 실제로 0~1 범위인지 확인합니다.
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

# 데이터셋 경로
DATASET_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax")
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def analyze_tiff_file(file_path: Path) -> dict:
    """단일 TIFF 파일 분석."""
    with Image.open(file_path) as img:
        mode = img.mode
        arr = np.array(img)

        return {
            "mode": mode,
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "in_range_0_1": bool(arr.min() >= 0 and arr.max() <= 1),
            "has_negative": bool(arr.min() < 0),
            "exceeds_1": bool(arr.max() > 1),
        }


def main():
    print("=" * 70)
    print("HDMAP TIFF 파일 값 범위 검증")
    print("=" * 70)
    print(f"\n데이터셋 경로: {DATASET_ROOT}")
    print(f"존재 여부: {DATASET_ROOT.exists()}")

    if not DATASET_ROOT.exists():
        print("ERROR: 데이터셋 경로가 존재하지 않습니다!")
        sys.exit(1)

    # 전체 통계
    all_stats = {
        "total_files": 0,
        "in_range_0_1": 0,
        "has_negative": 0,
        "exceeds_1": 0,
        "global_min": float("inf"),
        "global_max": float("-inf"),
    }

    # 도메인/split별 통계
    domain_stats = defaultdict(lambda: defaultdict(list))

    print("\n" + "-" * 70)
    print("도메인별 상세 분석")
    print("-" * 70)

    for domain in DOMAINS:
        domain_path = DATASET_ROOT / domain
        if not domain_path.exists():
            print(f"\n⚠️  {domain}: 경로 없음")
            continue

        print(f"\n📁 {domain}")

        # train/good
        train_path = domain_path / "train" / "good"
        if train_path.exists():
            tiff_files = list(train_path.glob("*.tiff"))
            print(f"   train/good: {len(tiff_files)} files")

            for f in tiff_files[:5]:  # 처음 5개만 상세 분석
                stats = analyze_tiff_file(f)
                domain_stats[domain]["train_good"].append(stats)
                all_stats["total_files"] += 1
                all_stats["global_min"] = min(all_stats["global_min"], stats["min"])
                all_stats["global_max"] = max(all_stats["global_max"], stats["max"])

                if stats["in_range_0_1"]:
                    all_stats["in_range_0_1"] += 1
                if stats["has_negative"]:
                    all_stats["has_negative"] += 1
                if stats["exceeds_1"]:
                    all_stats["exceeds_1"] += 1

            # 나머지 파일들도 통계에 포함
            for f in tiff_files[5:]:
                stats = analyze_tiff_file(f)
                all_stats["total_files"] += 1
                all_stats["global_min"] = min(all_stats["global_min"], stats["min"])
                all_stats["global_max"] = max(all_stats["global_max"], stats["max"])

                if stats["in_range_0_1"]:
                    all_stats["in_range_0_1"] += 1
                if stats["has_negative"]:
                    all_stats["has_negative"] += 1
                if stats["exceeds_1"]:
                    all_stats["exceeds_1"] += 1

        # test/good
        test_good_path = domain_path / "test" / "good"
        if test_good_path.exists():
            tiff_files = list(test_good_path.glob("*.tiff"))
            print(f"   test/good: {len(tiff_files)} files")

            for f in tiff_files[:5]:
                stats = analyze_tiff_file(f)
                domain_stats[domain]["test_good"].append(stats)
                all_stats["total_files"] += 1
                all_stats["global_min"] = min(all_stats["global_min"], stats["min"])
                all_stats["global_max"] = max(all_stats["global_max"], stats["max"])

                if stats["in_range_0_1"]:
                    all_stats["in_range_0_1"] += 1
                if stats["has_negative"]:
                    all_stats["has_negative"] += 1
                if stats["exceeds_1"]:
                    all_stats["exceeds_1"] += 1

            for f in tiff_files[5:]:
                stats = analyze_tiff_file(f)
                all_stats["total_files"] += 1
                all_stats["global_min"] = min(all_stats["global_min"], stats["min"])
                all_stats["global_max"] = max(all_stats["global_max"], stats["max"])

                if stats["in_range_0_1"]:
                    all_stats["in_range_0_1"] += 1
                if stats["has_negative"]:
                    all_stats["has_negative"] += 1
                if stats["exceeds_1"]:
                    all_stats["exceeds_1"] += 1

        # test/fault
        test_fault_path = domain_path / "test" / "fault"
        if test_fault_path.exists():
            tiff_files = list(test_fault_path.glob("*.tiff"))
            print(f"   test/fault: {len(tiff_files)} files")

            for f in tiff_files[:5]:
                stats = analyze_tiff_file(f)
                domain_stats[domain]["test_fault"].append(stats)
                all_stats["total_files"] += 1
                all_stats["global_min"] = min(all_stats["global_min"], stats["min"])
                all_stats["global_max"] = max(all_stats["global_max"], stats["max"])

                if stats["in_range_0_1"]:
                    all_stats["in_range_0_1"] += 1
                if stats["has_negative"]:
                    all_stats["has_negative"] += 1
                if stats["exceeds_1"]:
                    all_stats["exceeds_1"] += 1

            for f in tiff_files[5:]:
                stats = analyze_tiff_file(f)
                all_stats["total_files"] += 1
                all_stats["global_min"] = min(all_stats["global_min"], stats["min"])
                all_stats["global_max"] = max(all_stats["global_max"], stats["max"])

                if stats["in_range_0_1"]:
                    all_stats["in_range_0_1"] += 1
                if stats["has_negative"]:
                    all_stats["has_negative"] += 1
                if stats["exceeds_1"]:
                    all_stats["exceeds_1"] += 1

    # 샘플 파일 상세 정보 출력
    print("\n" + "-" * 70)
    print("샘플 파일 상세 정보 (각 split 첫 번째 파일)")
    print("-" * 70)

    for domain in DOMAINS:
        if domain not in domain_stats:
            continue
        print(f"\n📁 {domain}")
        for split, stats_list in domain_stats[domain].items():
            if stats_list:
                s = stats_list[0]
                status = "✅" if s["in_range_0_1"] else "❌"
                print(f"   {split}:")
                print(f"      mode: {s['mode']}, dtype: {s['dtype']}, shape: {s['shape']}")
                print(f"      min: {s['min']:.6f}, max: {s['max']:.6f}")
                print(f"      mean: {s['mean']:.6f}, std: {s['std']:.6f}")
                print(f"      범위 [0,1]: {status}")

    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)
    print(f"\n총 파일 수: {all_stats['total_files']}")
    print(f"전역 최솟값: {all_stats['global_min']:.6f}")
    print(f"전역 최댓값: {all_stats['global_max']:.6f}")
    print(f"\n범위 체크:")
    print(f"  [0,1] 범위 내: {all_stats['in_range_0_1']} / {all_stats['total_files']} ({100*all_stats['in_range_0_1']/all_stats['total_files']:.1f}%)")
    print(f"  음수 값 포함: {all_stats['has_negative']} / {all_stats['total_files']}")
    print(f"  1 초과 값 포함: {all_stats['exceeds_1']} / {all_stats['total_files']}")

    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    if all_stats["global_min"] >= 0 and all_stats["global_max"] <= 1:
        print("\n✅ 모든 TIFF 파일이 [0, 1] 범위 내에 있습니다.")
        print("   → np.clip(arr, 0, 1) * 255 변환이 적절합니다.")
    else:
        print(f"\n⚠️  일부 파일이 [0, 1] 범위를 벗어납니다!")
        print(f"   전역 범위: [{all_stats['global_min']:.6f}, {all_stats['global_max']:.6f}]")

        if all_stats["global_min"] < 0:
            print("   → 음수 값이 있으므로 클리핑 필요")
        if all_stats["global_max"] > 1:
            print("   → 1 초과 값이 있으므로 클리핑 또는 재정규화 필요")


if __name__ == "__main__":
    main()
