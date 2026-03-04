import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import os

def generate_ab_test_data():
    print("A/B 테스트 시뮬레이션 데이터 생성을 시작합니다...")
    
    # --- 1. 사전 통계 설계 (Power Analysis) ---
    baseline_cvr = 0.10      # 기존 전환율 (10%)
    lift = 0.12              # 목표 상승률 (상대적 12%)
    expected_cvr = baseline_cvr * (1 + lift)
    alpha = 0.05             # 유의 수준
    power = 0.8              # 검정력

    # Effect Size 및 필요한 샘플 사이즈 계산
    effect_size = proportion_effectsize(baseline_cvr, expected_cvr)
    analysis = NormalIndPower()
    required_n = math.ceil(analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0))
    
    print(f"통계적 유의성 확보를 위해 그룹당 {required_n}명의 샘플이 필요합니다.")

    # --- 2. 환경 설정 및 기본 데이터 틀 생성 ---
    np.random.seed(42)
    total_n = required_n * 2
    days = 14  # 2026년 3월 1일부터 14일간 실험 진행
    
    data = {
        'user_id': range(1, total_n + 1),
        'timestamp': [datetime(2026, 3, 1) + timedelta(days=np.random.randint(0, days), 
                                                     hours=np.random.randint(0, 24),
                                                     minutes=np.random.randint(0, 60)) for _ in range(total_n)],
        'group': ['A'] * required_n + ['B'] * required_n,
        'device': np.random.choice(['Mobile', 'PC'], total_n, p=[0.7, 0.3]),
        'source': np.random.choice(['Search', 'Social', 'Direct'], total_n, p=[0.4, 0.4, 0.2])
    }

    df = pd.DataFrame(data)
    # 그룹을 무작위로 섞음
    df = df.sample(frac=1).reset_index(drop=True)

    # --- 3. 현실적인 전환 로직 적용 (Realism Logic) ---
    def apply_conversion(row):
        # 기본 확률 설정
        prob = baseline_cvr
        
        # [변수 1] 실험군 효과 (가설 반영)
        if row['group'] == 'B':
            prob = expected_cvr
        
        # [변수 2] 기기별 노이즈 (모바일은 결제 단계가 번거로워 전환율이 낮음)
        if row['device'] == 'Mobile':
            prob -= 0.02
            
        # [변수 3] 요일별 노이즈 (주말에는 전환 의지가 낮아지는 경향 반영)
        if row['timestamp'].weekday() >= 5:
            prob -= 0.03
            
        # 최종 0 또는 1 결정
        return np.random.binomial(1, max(0, prob))

    df['converted'] = df.apply(apply_conversion, axis=1)

    # --- 4. 데이터 저장 ---
    # data 폴더가 없으면 생성
    if not os.path.exists('data'):
        os.makedirs('data')
        
    save_path = 'data/ab_test_data.csv'
    df.to_csv(save_path, index=False)
    
    print(f"데이터 생성 완료! 저장 경로: {save_path}")
    print(f"최종 생성된 샘플 수: {len(df)}개")
    print(f"그룹별 실제 전환율:\n{df.groupby('group')['converted'].mean()}")

if __name__ == "__main__":
    generate_ab_test_data()