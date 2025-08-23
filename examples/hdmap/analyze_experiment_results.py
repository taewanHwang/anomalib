#!/usr/bin/env python3
"""
실험 결과 분석 스크립트

이 스크립트는 여러 실험 세션의 결과를 분석하여 각 실험 조건별로
image_AUROC의 평균과 표준편차를 계산합니다.

사용법:
    python analyze_experiment_results.py --results_dir /path/to/results/draem
    python analyze_experiment_results.py --results_dir /path/to/results/draem --experiment_name "DRAEM_baseline_50epochs"
    uv run examples/hdmap/analyze_experiment_results.py --results_dir results/draem_sevnet
    uv run examples/hdmap/analyze_experiment_results.py --results_dir results_draem_14회/draem
    uv run examples/hdmap/analyze_experiment_results.py --results_dir results_draemsevnet_cond2/draem_sevnet
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


class ExperimentResultsAnalyzer:
    """실험 결과 분석기"""
    
    def __init__(self, results_dir: str, model_type: str = None):
        """
        Args:
            results_dir: 실험 결과가 저장된 기본 디렉토리 (예: results/draem)
            model_type: 모델 타입 ('draem', 'draem_sevnet', 'fastflow' 등). None이면 자동 감지
        """
        self.results_dir = Path(results_dir)
        self.model_type = model_type
        self.experiment_data = defaultdict(list)
        
        # 모델 타입 자동 감지
        if self.model_type is None:
            self.model_type = self._detect_model_type()
            print(f"자동 감지된 모델 타입: {self.model_type}")
        
    def _detect_model_type(self) -> str:
        """결과 디렉토리에서 모델 타입을 자동 감지합니다"""
        dir_name = self.results_dir.name.lower()
        
        # 디렉토리 이름에서 모델 타입 추출
        if 'draem_sevnet' in dir_name:
            return 'draem_sevnet'
        elif 'draem' in dir_name:
            return 'draem'
        elif 'fastflow' in dir_name:
            return 'fastflow'
        elif 'padim' in dir_name:
            return 'padim'
        else:
            # 기본값으로 draem 사용
            return 'draem'
        
    def find_all_experiment_sessions(self) -> List[Path]:
        """모든 실험 세션 폴더를 찾습니다 (타임스탬프 기반)"""
        sessions = []
        if not self.results_dir.exists():
            print(f"결과 디렉토리가 존재하지 않습니다: {self.results_dir}")
            return sessions
            
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.startswith('2025'):  # 타임스탬프 패턴
                sessions.append(item)
                
        return sorted(sessions, key=lambda x: x.name)
    
    def load_experiment_results(self, session_path: Path) -> Dict[str, dict]:
        """특정 세션의 모든 실험 결과를 로드합니다"""
        results = {}
        
        # MultiDomainHDMAP/{model_type}/ 하위의 모든 실험 폴더 검색
        experiment_base_path = session_path / "MultiDomainHDMAP" / self.model_type
        
        if not experiment_base_path.exists():
            return results
            
        for exp_folder in experiment_base_path.iterdir():
            if not exp_folder.is_dir():
                continue
                
            # tensorboard_logs/result_*.json 파일 찾기
            tensorboard_path = exp_folder / "tensorboard_logs"
            if not tensorboard_path.exists():
                continue
                
            for json_file in tensorboard_path.glob("result_*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 실험 이름 추출 - condition.name 우선, 없으면 experiment_name 사용
                    condition_info = data.get('condition', {})
                    if isinstance(condition_info, dict) and 'name' in condition_info:
                        condition_name = condition_info['name']
                    else:
                        condition_name = data.get('experiment_name', 'unknown')
                    
                    # 타임스탬프 패턴 제거 (예: _20250817_145220)
                    if condition_name.startswith('DRAEM_'):
                        condition_name = re.sub(r'_\d{8}_\d{6}$', '', condition_name)
                    
                    results[condition_name] = data
                    
                except Exception as e:
                    print(f"JSON 파일 로드 실패: {json_file}, 오류: {e}")
                    
        return results
    
    def collect_all_results(self) -> None:
        """모든 세션의 결과를 수집합니다"""
        sessions = self.find_all_experiment_sessions()
        print(f"발견된 실험 세션: {len(sessions)}개")
        
        for session in sessions:
            print(f"\n세션 분석 중: {session.name}")
            session_results = self.load_experiment_results(session)
            
            for condition_name, data in session_results.items():
                self.experiment_data[condition_name].append(data)
                
        print(f"\n총 수집된 실험 조건: {len(self.experiment_data)}개")
        for condition, runs in self.experiment_data.items():
            print(f"  {condition}: {len(runs)}회 실행")
    
    def calculate_statistics(self, experiment_name: str = None) -> pd.DataFrame:
        """통계를 계산합니다"""
        if experiment_name and experiment_name not in self.experiment_data:
            print(f"실험 '{experiment_name}'를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        conditions_to_analyze = [experiment_name] if experiment_name else list(self.experiment_data.keys())
        
        results = []
        
        for condition_name in conditions_to_analyze:
            runs = self.experiment_data[condition_name]
            if not runs:
                continue
                
            print(f"\n=== {condition_name} 분석 ===")
            print(f"총 실행 횟수: {len(runs)}")
            
            # 동적으로 source와 target 도메인 감지
            source_aurocs = []
            target_aurocs_by_domain = {}  # domain_name -> [auroc_values]
            avg_target_aurocs = []
            source_domain = None
            target_domains = []
            
            for run in runs:
                # Source domain 찾기
                source_result = run.get('source_results', {})
                if 'test_image_AUROC' in source_result:
                    source_aurocs.append(source_result['test_image_AUROC'])
                    # source 도메인 이름 추출 (condition.config에서)
                    if source_domain is None:
                        condition = run.get('condition', {})
                        config = condition.get('config', {})
                        if 'source_domain' in config:
                            source_domain = config['source_domain'].replace('domain_', '')  # domain_A -> A
                
                # Target domains 수집
                target_results = run.get('target_results', {})
                current_run_target_aucs = []
                
                for domain_key, domain_data in target_results.items():
                    if 'test_image_AUROC' in domain_data:
                        domain_name = domain_key.replace('domain_', '')  # domain_B -> B
                        
                        if domain_name not in target_aurocs_by_domain:
                            target_aurocs_by_domain[domain_name] = []
                            target_domains.append(domain_name)
                        
                        auroc_value = domain_data['test_image_AUROC']
                        target_aurocs_by_domain[domain_name].append(auroc_value)
                        current_run_target_aucs.append(auroc_value)
                
                # 이번 run의 평균 target AUROC 계산
                if current_run_target_aucs:
                    avg_target_aurocs.append(np.mean(current_run_target_aucs))
            
            # target_domains 정렬 (일관성을 위해)
            target_domains = sorted(target_domains)
            
            # 통계 계산 및 저장
            def calc_stats(values: List[float]) -> Tuple[float, float, int]:
                if not values:
                    return 0.0, 0.0, 0
                return np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0, len(values)
            
            source_mean, source_std, source_count = calc_stats(source_aurocs)
            avg_target_mean, avg_target_std, avg_target_count = calc_stats(avg_target_aurocs)
            
            # 각 target 도메인별 통계 계산
            target_stats = {}
            for domain in target_domains:
                if domain in target_aurocs_by_domain:
                    mean, std, count = calc_stats(target_aurocs_by_domain[domain])
                    target_stats[domain] = {
                        'mean': mean,
                        'std': std,
                        'count': count
                    }
            
            # Transfer ratio 계산 (avg_target_auroc_mean / source_auroc_mean)
            transfer_ratio = avg_target_mean / source_mean if source_mean > 0 else 0.0
            
            # 결과 딕셔너리 동적 구성 (experiment_name은 간단한 이름만)
            result_dict = {
                'experiment_name': condition_name,  # 간단한 실험 이름
                'total_runs': len(runs),
                'source_domain': source_domain or 'Unknown',
                'source_auroc_mean': source_mean,
                'source_auroc_std': source_std,
                'source_auroc_count': source_count,
                'avg_target_auroc_mean': avg_target_mean,
                'avg_target_auroc_std': avg_target_std,
                'avg_target_auroc_count': avg_target_count,
                'transfer_ratio': transfer_ratio
            }
            
            # 각 target 도메인별 결과 추가 (실제 target 도메인만)
            for domain in target_domains:
                if domain in target_stats:
                    result_dict[f'target_{domain}_auroc_mean'] = target_stats[domain]['mean']
                    result_dict[f'target_{domain}_auroc_std'] = target_stats[domain]['std']
                    result_dict[f'target_{domain}_auroc_count'] = target_stats[domain]['count']
            
            results.append(result_dict)
            
            # 콘솔 출력
            print(f"Source Domain: {source_domain or 'Unknown'}")
            print(f"Source AUROC: {source_mean:.4f} ± {source_std:.4f} (n={source_count})")
            
            # Target 도메인별 출력
            for domain in sorted(target_domains):
                if domain in target_stats:
                    stats = target_stats[domain]
                    print(f"Target {domain} AUROC: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            
            print(f"평균 Target AUROC: {avg_target_mean:.4f} ± {avg_target_std:.4f} (n={avg_target_count})")
            print(f"Transfer Ratio: {transfer_ratio:.4f}")
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str = None) -> None:
        """결과를 CSV 파일로 저장합니다"""
        if df.empty:
            print("저장할 결과가 없습니다.")
            return
            
        if output_path is None:
            output_path = self.results_dir / "experiment_analysis_summary.csv"
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n결과가 저장되었습니다: {output_path}")
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """요약 통계를 출력합니다"""
        if df.empty:
            print("출력할 결과가 없습니다.")
            return
            
        print("\n" + "="*80)
        print("실험 결과 요약")
        print("="*80)
        
        # 컬럼 너비 조정을 위한 설정
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        pd.set_option('display.expand_frame_repr', False)
        
        # 주요 컬럼만 선택해서 출력
        summary_cols = [
            'experiment_name', 'total_runs', 'source_domain',
            'source_auroc_mean', 'source_auroc_std',
            'avg_target_auroc_mean', 'avg_target_auroc_std',
            'transfer_ratio'
        ]
        
        summary_df = df[summary_cols].copy()
        
        # 정렬: source_auroc_mean 높은 순, 같다면 avg_target_auroc_mean 높은 순
        summary_df = summary_df.sort_values(
            by=['source_auroc_mean', 'avg_target_auroc_mean'], 
            ascending=[False, False]
        )
        
        summary_df['source_auroc_mean'] = summary_df['source_auroc_mean'].round(4)
        summary_df['source_auroc_std'] = summary_df['source_auroc_std'].round(4)
        summary_df['avg_target_auroc_mean'] = summary_df['avg_target_auroc_mean'].round(4)
        summary_df['avg_target_auroc_std'] = summary_df['avg_target_auroc_std'].round(4)
        summary_df['transfer_ratio'] = summary_df['transfer_ratio'].round(4)
        
        print(summary_df.to_string(index=False))
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='실험 결과 분석')
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='/home/taewan.hwang/study/anomalib/results/draem',
        help='실험 결과 디렉토리 경로'
    )
    parser.add_argument(
        '--model_type', 
        type=str, 
        default=None,
        help='모델 타입 (draem, draem_sevnet, fastflow, padim 등). None이면 자동 감지'
    )
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default=None,
        help='특정 실험만 분석 (예: DRAEM_baseline_50epochs)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='결과 저장 파일 경로 (기본: results_dir/experiment_analysis_summary.csv)'
    )
    
    args = parser.parse_args()
    
    print(f"실험 결과 분석 시작...")
    print(f"결과 디렉토리: {args.results_dir}")
    if args.model_type:
        print(f"지정된 모델 타입: {args.model_type}")
    if args.experiment_name:
        print(f"특정 실험 분석: {args.experiment_name}")
    
    # 분석기 초기화 및 실행
    analyzer = ExperimentResultsAnalyzer(args.results_dir, args.model_type)
    analyzer.collect_all_results()
    
    # 통계 계산
    results_df = analyzer.calculate_statistics(args.experiment_name)
    
    # 결과 출력 및 저장
    analyzer.print_summary(results_df)
    analyzer.save_results(results_df, args.output)


if __name__ == "__main__":
    main()
