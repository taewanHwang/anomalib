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
   examples/hdmap/analyze_experiment_results.py --results_dir results4 --all-models

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

                    # Source AUROC 추출 (multi-domain 형식)
                    source_auroc = None
                    source_domain = None
                    if 'source_results' in data:
                        if 'auroc' in data['source_results']:
                            source_auroc = data['source_results']['auroc']
                        elif 'test_image_AUROC' in data['source_results']:
                            source_auroc = data['source_results']['test_image_AUROC']

                        if 'domain' in data['source_results']:
                            source_domain = data['source_results']['domain']

                    # 실험 설정에서 source domain 추출 (fallback)
                    if source_domain is None and 'source_domain' in data:
                        source_domain = data['source_domain']

                    # Target AUROCs 수집
                    target_aurocs = []
                    target_domains = []

                    # 각 target domain 결과를 개별 행으로 추가
                    for domain_key, domain_data in data['target_results'].items():
                        auroc_value = None
                        if 'auroc' in domain_data:
                            auroc_value = domain_data['auroc']
                        elif 'test_image_AUROC' in domain_data:
                            auroc_value = domain_data['test_image_AUROC']

                        if auroc_value is not None:
                            domain_name = domain_key.replace('domain_', '')
                            target_aurocs.append(auroc_value)
                            target_domains.append(domain_name)

                            experiment_results.append({
                                'experiment_name': exp_name,
                                'source_domain': source_domain,
                                'source_AUROC': source_auroc,
                                'target_AUROC': auroc_value,
                                'target_domain': domain_name,
                                'session_id': session_id,
                                'type': 'Multi-domain',
                                'severity_input_channels': data.get('config', {}).get('severity_input_channels', 'N/A')
                            })

                    # 평균 target AUROC 계산 및 추가
                    if target_aurocs:
                        avg_target_auroc = sum(target_aurocs) / len(target_aurocs)
                        experiment_results.append({
                            'experiment_name': exp_name,
                            'source_domain': source_domain,
                            'source_AUROC': source_auroc,
                            'target_AUROC': avg_target_auroc,
                            'target_domain': 'Average',
                            'session_id': session_id,
                            'type': 'Multi-domain',
                            'severity_input_channels': data.get('config', {}).get('severity_input_channels', 'N/A')
                        })
                
                else:
                    # Single domain 실험
                    source_auroc = None
                    source_domain = None

                    if 'source_results' in data:
                        if 'auroc' in data['source_results']:
                            source_auroc = data['source_results']['auroc']
                        elif 'test_image_AUROC' in data['source_results']:
                            source_auroc = data['source_results']['test_image_AUROC']

                        if 'domain' in data['source_results']:
                            source_domain = data['source_results']['domain']

                    # 실험 설정에서 source domain 추출 (fallback)
                    if source_domain is None and 'source_domain' in data:
                        source_domain = data['source_domain']

                    if source_auroc is not None:
                        experiment_results.append({
                            'experiment_name': exp_dir.name,
                            'source_domain': source_domain,
                            'source_AUROC': source_auroc,
                            'target_AUROC': 'N/A',
                            'target_domain': 'N/A',
                            'session_id': exp_dir.parent.name,
                            'type': 'Single-domain',
                            'severity_input_channels': data.get('config', {}).get('severity_input_channels', 'N/A')
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

    # Multi-domain 실험 전용 분석 추가
    multi_domain_df = combined_df[combined_df['type'] == 'Multi-domain'].copy()

    if not multi_domain_df.empty:
        print(f"\n{'='*80}")
        print(f"🌍 Multi-Domain 실험 상세 분석")
        print(f"{'='*80}")

        # Multi-domain 실험의 평균 결과만 추출
        multi_avg_df = multi_domain_df[multi_domain_df['target_domain'] == 'Average'].copy()

        if not multi_avg_df.empty:
            print(f"\n📊 Multi-Domain 실험별 평균 성능 (Source → Target Mean AUROC):")
            print("-" * 80)

            # 실험명에서 설정 정보 추출 및 정렬
            multi_avg_df_sorted = multi_avg_df.sort_values(['source_domain', 'target_AUROC'], ascending=[True, False])

            print(f"{'실험명':<45} {'Source':<8} {'Src AUROC':<10} {'Tgt Avg':<10} {'Severity Ch':<15}")
            print("-" * 95)

            for _, row in multi_avg_df_sorted.iterrows():
                exp_name = row['experiment_name'][:40] + "..." if len(row['experiment_name']) > 40 else row['experiment_name']
                source_dom = str(row['source_domain']).replace('domain_', '') if row['source_domain'] else 'N/A'
                src_auroc = f"{row['source_AUROC']:.6f}" if row['source_AUROC'] is not None else 'N/A'
                tgt_auroc = f"{row['target_AUROC']:.6f}" if row['target_AUROC'] != 'N/A' else 'N/A'
                severity_ch = str(row['severity_input_channels'])[:12] + "..." if len(str(row['severity_input_channels'])) > 12 else str(row['severity_input_channels'])

                print(f"{exp_name:<45} {source_dom:<8} {src_auroc:<10} {tgt_auroc:<10} {severity_ch:<15}")

        # Source domain별 성능 요약
        print(f"\n📈 Source Domain별 성능 요약:")
        print("-" * 60)

        source_summary = multi_avg_df.groupby('source_domain').agg({
            'source_AUROC': ['mean', 'std', 'count'],
            'target_AUROC': ['mean', 'std', 'count']
        }).round(6)

        source_summary.columns = ['Src_Mean', 'Src_Std', 'Src_Count', 'Tgt_Mean', 'Tgt_Std', 'Tgt_Count']
        source_summary = source_summary.sort_values('Tgt_Mean', ascending=False)

        print(f"{'Source':<8} {'Src AUROC':<20} {'Target Avg AUROC':<20} {'실험수':<8}")
        print("-" * 60)

        for source_dom, row in source_summary.iterrows():
            source_dom_short = str(source_dom).replace('domain_', '') if source_dom else 'N/A'
            src_perf = f"{row['Src_Mean']:.4f}±{row['Src_Std']:.4f}" if pd.notna(row['Src_Std']) else f"{row['Src_Mean']:.4f}"
            tgt_perf = f"{row['Tgt_Mean']:.4f}±{row['Tgt_Std']:.4f}" if pd.notna(row['Tgt_Std']) else f"{row['Tgt_Mean']:.4f}"
            exp_count = int(row['Src_Count'])

            print(f"{source_dom_short:<8} {src_perf:<20} {tgt_perf:<20} {exp_count:<8}")

        # Severity input channels별 성능 요약
        severity_summary = multi_avg_df.groupby('severity_input_channels').agg({
            'source_AUROC': ['mean', 'std', 'count'],
            'target_AUROC': ['mean', 'std', 'count']
        }).round(6)

        severity_summary.columns = ['Src_Mean', 'Src_Std', 'Src_Count', 'Tgt_Mean', 'Tgt_Std', 'Tgt_Count']
        severity_summary = severity_summary.sort_values('Tgt_Mean', ascending=False)

        print(f"\n🔧 Severity Input Channels별 성능 요약:")
        print("-" * 70)
        print(f"{'Severity Channels':<20} {'Src AUROC':<20} {'Target Avg AUROC':<20} {'실험수':<8}")
        print("-" * 70)

        for severity_ch, row in severity_summary.iterrows():
            severity_ch_str = str(severity_ch)[:18] + ".." if len(str(severity_ch)) > 18 else str(severity_ch)
            src_perf = f"{row['Src_Mean']:.4f}±{row['Src_Std']:.4f}" if pd.notna(row['Src_Std']) else f"{row['Src_Mean']:.4f}"
            tgt_perf = f"{row['Tgt_Mean']:.4f}±{row['Tgt_Std']:.4f}" if pd.notna(row['Tgt_Std']) else f"{row['Tgt_Mean']:.4f}"
            exp_count = int(row['Src_Count'])

            print(f"{severity_ch_str:<20} {src_perf:<20} {tgt_perf:<20} {exp_count:<8}")

        # Target domain별 상세 분석 (개별 target domain 성능)
        multi_targets_df = multi_domain_df[multi_domain_df['target_domain'] != 'Average'].copy()

        if not multi_targets_df.empty:
            print(f"\n🎯 Target Domain별 상세 성능 분석:")
            print("-" * 80)

            target_perf = multi_targets_df.groupby(['source_domain', 'target_domain']).agg({
                'target_AUROC': ['mean', 'std', 'count']
            }).round(6)

            target_perf.columns = ['Mean', 'Std', 'Count']
            target_perf = target_perf.sort_values('Mean', ascending=False)

            print(f"{'Source → Target':<20} {'AUROC':<20} {'실험수':<8}")
            print("-" * 50)

            for (src, tgt), row in target_perf.iterrows():
                src_short = str(src).replace('domain_', '') if src else 'N/A'
                tgt_short = str(tgt).replace('domain_', '') if tgt else 'N/A'
                transfer = f"{src_short} → {tgt_short}"
                perf = f"{row['Mean']:.4f}±{row['Std']:.4f}" if pd.notna(row['Std']) else f"{row['Mean']:.4f}"
                exp_count = int(row['Count'])

                print(f"{transfer:<20} {perf:<20} {exp_count:<8}")

        # Multi-domain 실험 결과 매트릭스 생성
        if not multi_avg_df.empty:
            print(f"\n📋 Multi-Domain 성능 매트릭스 (Source → Target Average AUROC):")
            print("-" * 80)

            # 피벗 테이블 생성
            pivot_matrix = multi_avg_df.pivot_table(
                values='target_AUROC',
                index=['source_domain', 'severity_input_channels'],
                columns='target_domain',
                fill_value=None
            )

            # 매트릭스 출력
            if 'Average' in pivot_matrix.columns:
                # Source domain과 severity channels로 그룹화하여 출력
                print(f"{'Source_Severity':<25} {'Avg AUROC':<12} {'Src AUROC':<12}")
                print("-" * 50)

                for (src_dom, severity_ch), row in pivot_matrix.iterrows():
                    if pd.notna(row['Average']):
                        src_short = str(src_dom).replace('domain_', '') if src_dom else 'N/A'
                        severity_short = str(severity_ch)[:12] + ".." if len(str(severity_ch)) > 12 else str(severity_ch)
                        src_severity = f"{src_short}_{severity_short}"

                        # 해당 source AUROC 찾기
                        src_auroc_row = multi_avg_df[
                            (multi_avg_df['source_domain'] == src_dom) &
                            (multi_avg_df['severity_input_channels'] == severity_ch)
                        ]
                        src_auroc = src_auroc_row['source_AUROC'].iloc[0] if not src_auroc_row.empty else 'N/A'
                        src_auroc_str = f"{src_auroc:.6f}" if src_auroc != 'N/A' else 'N/A'

                        print(f"{src_severity:<25} {row['Average']:<12.6f} {src_auroc_str:<12}")

        # Multi-domain CSV 저장
        multi_domain_summary_path = Path(output).parent / "multi_domain_analysis.csv" if output else results_base_path / "multi_domain_analysis.csv"
        multi_domain_df.to_csv(multi_domain_summary_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Multi-domain 분석 결과 저장됨: {multi_domain_summary_path}")
    
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