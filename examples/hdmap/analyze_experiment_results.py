#!/usr/bin/env python3
"""
실험 결과 분석 스크립트

이 스크립트는 여러 실험 세션의 결과를 분석하여 각 실험 조건별로
image_AUROC의 평균과 표준편차를 계산합니다.

==============================================================================
🚀 기본 사용법:
==============================================================================

1. 단일 모델 분석:
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results/draem_single
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results/dinomaly_single
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results/patchcore_single

2. 모든 모델 통합 분석 (추천):
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models

3. 특정 실험 조건만 분석:
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models --experiment_name "baseline"

4. 결과 CSV로 저장:
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models --output comparison.csv

==============================================================================
📁 대상 폴더 구조 (base-run.sh 결과):
==============================================================================

results/
├── draem_single/20250830_143052/SingleDomainHDMAP/DRAEM/...
├── dinomaly_single/20250830_143052/SingleDomainHDMAP/Dinomaly/...  
├── patchcore_single/20250830_143052/SingleDomainHDMAP/PatchCore/...
└── draem_sevnet_single/20250830_143052/SingleDomainHDMAP/DRAEM_SevNet/...

==============================================================================
📊 출력 내용:
==============================================================================

--all-models 사용 시:
- 모델별 평균/최고/최저 AUROC 요약
- 전체 실험 조건별 상세 성능 (AUROC 순 정렬)  
- CSV 파일 자동 생성:
  * all_models_analysis_summary.csv (전체 상세 결과)
  * models_summary_all_models_analysis.csv (모델별 요약)

==============================================================================
🔧 고급 옵션:
==============================================================================

--model_type: 모델 타입 명시 (draem, dinomaly, patchcore 등)
--experiment_name: 특정 실험만 분석 (부분 문자열 매칭)
--output: 결과 CSV 저장 경로 지정

==============================================================================
💡 이전 버전 호환성:
==============================================================================

기존 multidomain 결과도 분석 가능:
    .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results_draem_14회/draem
    .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results_patchcore_AtoD/patchcore
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
        
        # 디렉토리 이름에서 모델 타입 추출 (대문자 반환)
        if 'draem_sevnet' in dir_name:
            return 'DRAEM_SevNet'
        elif 'draem' in dir_name:
            return 'DRAEM'
        elif 'dinomaly' in dir_name:
            return 'Dinomaly'
        elif 'fastflow' in dir_name:
            return 'FastFlow'
        elif 'padim' in dir_name:
            return 'Padim'
        elif 'patchcore' in dir_name:
            return 'PatchCore'
        else:
            # 기본값으로 DRAEM 사용
            return 'DRAEM'
        
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
        
        # MultiDomainHDMAP/{model_type}/ 또는 SingleDomainHDMAP/{model_type}/ 하위의 모든 실험 폴더 검색
        possible_paths = [
            session_path / "MultiDomainHDMAP" / self.model_type,
            session_path / "SingleDomainHDMAP" / self.model_type
        ]
        
        experiment_base_path = None
        for path in possible_paths:
            if path.exists():
                experiment_base_path = path
                break
        
        if experiment_base_path is None:
            print(f"실험 결과 경로를 찾을 수 없습니다: {possible_paths}")
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
                # Source domain 찾기 - source_results에서 먼저 찾고, 없으면 results에서 찾기 (단일 도메인 실험용)
                source_result = run.get('source_results', {})
                auroc_value = None
                
                if 'test_image_AUROC' in source_result:
                    auroc_value = source_result['test_image_AUROC']
                elif 'image_AUROC' in run.get('results', {}):
                    # 단일 도메인 실험의 경우 results.image_AUROC 사용
                    auroc_value = run.get('results', {})['image_AUROC']
                elif 'test_image_AUROC' in run.get('source_results', {}):
                    auroc_value = run.get('source_results', {})['test_image_AUROC']
                
                if auroc_value is not None:
                    source_aurocs.append(auroc_value)
                    # source 도메인 이름 추출 (condition.config에서)
                    if source_domain is None:
                        condition = run.get('condition', {})
                        config = condition.get('config', {})
                        if 'source_domain' in config:
                            source_domain = config['source_domain'].replace('domain_', '')  # domain_A -> A
                        elif 'domain' in run.get('results', {}):
                            # 단일 도메인 실험의 경우 results.domain 사용
                            source_domain = run.get('results', {})['domain'].replace('domain_', '')
                
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


def analyze_all_models(results_base_dir: str, experiment_name: str = None, output: str = None):
    """모든 모델의 결과를 통합 분석"""
    results_base_path = Path(results_base_dir)
    
    print(f"🔍 모든 모델 통합 분석 시작...")
    print(f"📁 기본 디렉토리: {results_base_path}")
    
    # 모델별 디렉토리 찾기 (*_single 패턴)
    model_dirs = list(results_base_path.glob("*_single"))
    if not model_dirs:
        print(f"❌ {results_base_path}에서 *_single 패턴의 모델 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"📊 발견된 모델: {[d.name for d in model_dirs]}")
    
    all_results = []
    
    for model_dir in sorted(model_dirs):
        model_type_lower = model_dir.name.replace('_single', '')
        # 실제 폴더명에 맞는 모델 타입 매핑 (대문자)
        model_type_mapping = {
            'draem': 'DRAEM',
            'dinomaly': 'Dinomaly', 
            'patchcore': 'PatchCore',
            'draem_sevnet': 'DRAEM_SevNet'
        }
        model_type = model_type_mapping.get(model_type_lower, model_type_lower.upper())
        
        print(f"\n🔬 {model_type} 분석 중...")
        
        try:
            # 각 모델별 분석
            analyzer = ExperimentResultsAnalyzer(str(model_dir), model_type)
            analyzer.collect_all_results()
            
            if not analyzer.experiment_data:
                print(f"⚠️ {model_type}에서 유효한 실험 데이터를 찾을 수 없습니다.")
                continue
                
            # 통계 계산
            model_df = analyzer.calculate_statistics(experiment_name)
            
            if not model_df.empty:
                # 모델 타입 컬럼 추가
                model_df['Model'] = model_type
                all_results.append(model_df)
                print(f"✅ {model_type}: {len(model_df)} 개 실험 조건")
            else:
                print(f"⚠️ {model_type}: 분석할 데이터가 없습니다.")
                
        except Exception as e:
            print(f"❌ {model_type} 분석 실패: {e}")
            continue
    
    if not all_results:
        print("❌ 분석할 수 있는 모델 결과가 없습니다.")
        return
    
    # 모든 결과 통합
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 컬럼 순서 재정렬 (Model을 앞으로)
    cols = ['Model'] + [col for col in combined_df.columns if col != 'Model']
    combined_df = combined_df[cols]
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"🎯 모든 모델 통합 분석 결과")
    print(f"{'='*80}")
    print(f"총 모델 수: {combined_df['Model'].nunique()}")
    print(f"총 실험 조건 수: {len(combined_df)}")
    print(f"\n📊 모델별 Image AUROC 요약:")
    
    # 모델별 평균 성능 출력
    auroc_column = 'source_auroc_mean'  # 실제 컬럼명 사용
    if auroc_column in combined_df.columns:
        model_summary = combined_df.groupby('Model')[auroc_column].agg(['mean', 'max', 'min', 'count']).round(4)
        model_summary.columns = ['평균_AUROC', '최고_AUROC', '최저_AUROC', '실험_수']
        model_summary = model_summary.sort_values('평균_AUROC', ascending=False)
        
        print(model_summary)
        
        print(f"\n📈 전체 상세 결과:")
        # Image AUROC 기준으로 정렬해서 출력
        display_df = combined_df.sort_values(auroc_column, ascending=False)
    else:
        print(f"⚠️ AUROC 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(combined_df.columns)}")
        display_df = combined_df
    print(display_df.to_string(index=False))
    
    # 결과 저장
    if output is None:
        output = results_base_path / "all_models_analysis_summary.csv"
    else:
        output = Path(output)
    
    combined_df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"\n💾 통합 결과 저장됨: {output}")
    
    # 모델별 요약도 저장
    summary_output = output.parent / f"models_summary_{output.stem}.csv"
    model_summary.to_csv(summary_output, encoding='utf-8-sig')
    print(f"📋 모델별 요약 저장됨: {summary_output}")


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
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='results 디렉토리의 모든 모델 결과를 통합 분석 (예: results/)'
    )
    
    args = parser.parse_args()
    
    if args.all_models:
        # 모든 모델 통합 분석
        analyze_all_models(args.results_dir, args.experiment_name, args.output)
    else:
        # 단일 모델 분석 (기존 방식)
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
