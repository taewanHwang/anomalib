#!/usr/bin/env python3
"""
실험 결과 분석 스크립트

이 스크립트는 여러 실험의 결과를 분석하여 각 실험별로
image_AUROC 값을 추출합니다.

==============================================================================
🚀 기본 사용법:
==============================================================================

1. 모든 실험 분석:
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models

2. 결과 CSV로 저장:
   .venv/bin/python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models --output comparison.csv

==============================================================================
📁 대상 폴더 구조 (통합된 저장 방식):
==============================================================================

results/
└── 20250831_074352/
    ├── domainA_to_BCD_draem_quick_test_20250831_074352/
    │   └── result_20250831_080624.json
    ├── domainA_patchcore_baseline_20250831_074352/
    │   └── result_20250831_081234.json
    └── ...

==============================================================================
📊 출력 내용:
==============================================================================

- 전체 실험별 AUROC 값 (높은 순으로 정렬)
- CSV 파일 자동 생성:
  * experiment_analysis_summary.csv (전체 실험 결과)

==============================================================================
🔧 고급 옵션:
==============================================================================

--output: 결과 CSV 저장 경로 지정
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_all_models(results_base_dir: str, output: str = None):
    """모든 실험의 결과를 분석 (단순화된 버전)"""
    results_base_path = Path(results_base_dir)
    
    print(f"🔍 실험 결과 분석 시작...")
    print(f"📁 기본 디렉토리: {results_base_path}")
    
    # timestamp 디렉토리를 찾고, 그 안의 실험 디렉토리들을 분석
    timestamp_dirs = [d for d in results_base_path.iterdir() if d.is_dir() and d.name.replace('_', '').isdigit()]
    
    if not timestamp_dirs:
        print(f"❌ {results_base_path}에서 timestamp 디렉토리를 찾을 수 없습니다.")
        return
    
    # 모든 실험 디렉토리를 수집
    all_experiment_dirs = []
    for timestamp_dir in timestamp_dirs:
        experiment_dirs = [d for d in timestamp_dir.iterdir() if d.is_dir()]
        all_experiment_dirs.extend(experiment_dirs)
    
    if not all_experiment_dirs:
        print(f"❌ timestamp 디렉토리들에서 실험 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"📊 발견된 실험 디렉토리: {len(all_experiment_dirs)}개")
    print(f"   예시: {[d.name for d in all_experiment_dirs[:3]]}")
    
    # 각 실험 디렉토리에서 결과 수집
    experiment_results = []
    
    for exp_dir in all_experiment_dirs:
        # 실험 루트 디렉토리에서 result_*.json 파일 찾기 (통합된 저장 방식)
        json_files = list(exp_dir.glob("result_*.json"))
        
        if json_files:
            try:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Single domain vs Multi domain 구분
                if 'target_results' in data and data['target_results']:
                    # Multi-domain 실험
                    exp_name = exp_dir.name
                    session_id = exp_dir.parent.name
                    
                    source_auroc = None
                    if 'source_results' in data and 'test_image_AUROC' in data['source_results']:
                        source_auroc = data['source_results']['test_image_AUROC']
                    
                    # 각 target domain 결과를 개별 행으로 추가
                    for domain_key, domain_data in data['target_results'].items():
                        if 'test_image_AUROC' in domain_data:
                            domain_name = domain_key.replace('domain_', '')
                            experiment_results.append({
                                'experiment_name': exp_name,
                                'source_AUROC': source_auroc,
                                'target_AUROC': domain_data['test_image_AUROC'],
                                'target_domain': domain_name,
                                'session_id': session_id,
                                'type': 'Multi-domain'
                            })
                    
                    # 평균 target AUROC도 추가
                    if 'avg_target_auroc' in data:
                        experiment_results.append({
                            'experiment_name': exp_name,
                            'source_AUROC': source_auroc,
                            'target_AUROC': data['avg_target_auroc'],
                            'target_domain': 'Average',
                            'session_id': session_id,
                            'type': 'Multi-domain'
                        })
                
                else:
                    # Single domain 실험
                    source_auroc = None
                    if 'source_results' in data and 'test_image_AUROC' in data['source_results']:
                        source_auroc = data['source_results']['test_image_AUROC']
                    
                    if source_auroc is not None:
                        experiment_results.append({
                            'experiment_name': exp_dir.name,
                            'source_AUROC': source_auroc,
                            'target_AUROC': 'N/A',
                            'target_domain': 'N/A',
                            'session_id': exp_dir.parent.name,
                            'type': 'Single-domain'
                        })
                    else:
                        print(f"   ⚠️ {exp_dir.name}에서 AUROC 값을 찾을 수 없습니다.")
                        
            except Exception as e:
                print(f"   ⚠️ {exp_dir.name} JSON 파싱 오류: {e}")
        else:
            print(f"   ⚠️ {exp_dir.name}에서 result JSON 파일을 찾을 수 없습니다.")
    
    if not experiment_results:
        print("❌ 분석할 수 있는 실험 결과가 없습니다.")
        return
    
    # DataFrame 생성
    combined_df = pd.DataFrame(experiment_results)
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"🎯 실험 분석 결과")
    print(f"{'='*80}")
    print(f"총 실험 수: {len(combined_df)}")
    
    # target_AUROC 기준으로 정렬 (N/A가 아닌 것만), 그 다음 source_AUROC
    # N/A를 맨 아래로 보내기 위해 숫자 변환
    display_df = combined_df.copy()
    display_df['sort_target'] = display_df['target_AUROC'].apply(lambda x: -1 if x == 'N/A' else float(x))
    display_df = display_df.sort_values(['sort_target', 'source_AUROC'], ascending=[False, False]).drop('sort_target', axis=1)
    
    print(f"\n📈 전체 실험 결과:")
    print(display_df.to_string(index=False))
    
    # 도메인별 평균과 실험 조건별 전체 평균 분석 (Single-domain 실험만)
    single_domain_df = combined_df[combined_df['type'] == 'Single-domain'].copy()
    
    if not single_domain_df.empty:
        print(f"\n{'='*80}")
        print(f"📊 실험 조건별 평균 성능 분석 (Single-domain)")
        print(f"{'='*80}")
        
        # 실험 이름에서 도메인과 조건 추출
        def extract_condition_and_domain(exp_name):
            parts = exp_name.split('_')
            if len(parts) >= 2:
                domain = parts[0]  # domainA, domainB, etc.
                # timestamp 제거 (마지막 부분이 숫자로만 이루어진 경우)
                condition_parts = []
                for part in parts[1:]:
                    if part.replace('_', '').isdigit() and len(part) >= 8:  # timestamp 형태
                        break
                    condition_parts.append(part)
                condition = '_'.join(condition_parts)
                return domain, condition
            return None, None
        
        single_domain_df['domain'] = single_domain_df['experiment_name'].apply(lambda x: extract_condition_and_domain(x)[0])
        single_domain_df['condition'] = single_domain_df['experiment_name'].apply(lambda x: extract_condition_and_domain(x)[1])
        
        # 유효한 도메인과 조건만 필터링
        valid_df = single_domain_df.dropna(subset=['domain', 'condition'])
        
        if not valid_df.empty:
            # 조건별 전체 평균 (모든 도메인과 실행에 대한 평균)
            condition_avg = valid_df.groupby('condition')['source_AUROC'].agg(['mean', 'std', 'count']).reset_index()
            condition_avg.columns = ['condition', 'avg_AUROC', 'std_AUROC', 'experiment_count']
            condition_avg = condition_avg.sort_values('avg_AUROC', ascending=False)
            
            print(f"\n🎯 실험 조건별 전체 평균 (모든 도메인, 모든 실행):")
            print(f"{'조건':<50} {'평균 AUROC':<12} {'표준편차':<10} {'실험 수':<8}")
            print("-" * 82)
            for _, row in condition_avg.iterrows():
                std_str = f"±{row['std_AUROC']:.4f}" if pd.notna(row['std_AUROC']) else "N/A"
                print(f"{row['condition']:<50} {row['avg_AUROC']:<12.6f} {std_str:<10} {int(row['experiment_count']):<8}")
            
            # 조건별, 도메인별 평균
            domain_condition_avg = valid_df.groupby(['condition', 'domain'])['source_AUROC'].agg(['mean', 'std', 'count']).reset_index()
            domain_condition_avg.columns = ['condition', 'domain', 'avg_AUROC', 'std_AUROC', 'experiment_count']
            
            # 조건별로 도메인 결과를 피벗
            pivot_df = domain_condition_avg.pivot_table(
                values='avg_AUROC', 
                index='condition', 
                columns='domain', 
                fill_value=None
            )
            
            # 전체 평균과 함께 표시
            summary_df = condition_avg.set_index('condition')[['avg_AUROC']].copy()
            summary_df.columns = ['Overall_Avg']
            
            # 도메인별 결과와 전체 평균 결합
            final_df = pd.concat([pivot_df, summary_df], axis=1)
            final_df = final_df.sort_values('Overall_Avg', ascending=False)
            
            print(f"\n📋 실험 조건별 도메인 성능 매트릭스:")
            print("=" * 100)
            
            # 컬럼 헤더 출력
            domains = [col for col in final_df.columns if col.startswith('domain')]
            header = f"{'조건':<50}"
            for domain in sorted(domains):
                header += f" {domain:<10}"
            header += f" {'전체평균':<10}"
            print(header)
            print("-" * len(header))
            
            # 각 행 출력
            for condition, row in final_df.iterrows():
                line = f"{condition:<50}"
                for domain in sorted(domains):
                    if domain in row and pd.notna(row[domain]):
                        line += f" {row[domain]:<10.6f}"
                    else:
                        line += f" {'N/A':<10}"
                line += f" {row['Overall_Avg']:<10.6f}"
                print(line)
            
            # CSV로도 저장
            condition_summary_path = Path(output).parent / "experiment_condition_summary.csv" if output else results_base_path / "experiment_condition_summary.csv"
            final_df.to_csv(condition_summary_path, encoding='utf-8-sig')
            print(f"\n💾 조건별 요약 저장됨: {condition_summary_path}")
        
        else:
            print("⚠️ 유효한 single-domain 실험 데이터가 없습니다.")
    
    # 결과 저장
    if output is None:
        output = results_base_path / "experiment_analysis_summary.csv"
    else:
        output = Path(output)
    
    combined_df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장됨: {output}")


def main():
    parser = argparse.ArgumentParser(description='실험 결과 분석')
    parser.add_argument(
        '--results_dir', 
        type=str, 
        default='results',
        help='실험 결과 디렉토리 경로 (기본: results)'
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
        help='모든 실험 결과를 분석 (필수 옵션)'
    )
    
    args = parser.parse_args()
    
    if not args.all_models:
        print("❌ --all-models 옵션을 사용하여 분석을 수행하세요.")
        print("예: python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models")
        return
    
    # 통합 분석 실행
    analyze_all_models(args.results_dir, args.output)


if __name__ == "__main__":
    main()