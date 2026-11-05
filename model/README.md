# model/ — A 차선: 가상 손보사 모형

`interfaces.contracts.KicsModel`을 구현한다. 공시(2026-06-30)로 복원한 가상 보험사 4곳을 제공한다.

| 키 | 기반 | 기초 K-ICS | 기본자본비율 |
|---|---|---|---|
| A | 삼성화재 | 282.8% (공시와 일치) | 220% |
| B | DB손해보험 | 204.4% (공시와 일치) | 96% |
| C | 상위 5개사 합계 | 231.4% (5개사 합계와 일치) | 132% |
| D | 비상위 6개사 합계, 자본 비례 조정 | 150.0% (설정값, 관측치 아님) | 62% |

## 쓰는 법

```python
import numpy as np
from interfaces.contracts import N_FACTORS, NO_ACTION, Regime
from model.archetypes import build_all
from model.insurer import StylizedInsurer

models = {k: StylizedInsurer(p) for k, p in build_all().items()}
m = models["B"]
b = m.initial()
for year in range(3):
    b = m.step(b, np.zeros(N_FACTORS), Regime.NONE, NO_ACTION)
print(b.ratio, b.basic_ratio)
```

- `step()`은 `BalanceSheet`의 하위 클래스 `InsurerState`를 돌려준다. 위험액 구성, 자본 계층, 대응으로 줄어든 노출, 누적 계리가정 변화를 함께 싣는다. **다음 해에는 받은 상태를 그대로 넘겨야** 이월이 정확하다. 일반 `BalanceSheet`를 넘기면 기준 시점 구성비로 복원한다.
- `basic_ratio`는 기본자본 K-ICS 비율이다(2027년 50% 기준).
- D의 기초 비율은 `build_all(target_ratio_d=1.40)`처럼 바꿀 수 있다.
- 한 번 이월은 0.6~1.1ms다(같은 계리가정 조합은 캐시). 3년 × 점프 패턴 16 × 다중시작 수백 회가 수십 초 수준이다.

## 상태 필드 (`InsurerState`)

| 필드 | 뜻 |
|---|---|
| `available_capital`, `required_capital`, `ratio`, `basic_ratio` | 가용·요구자본(조원), K-ICS 비율, 기본자본비율 |
| `risks` | `RiskCharges(insurance_long, insurance_general, market, credit)` 대분류 위험액 (조원) |
| `basic_capital`, `supplementary_capital`, `supplementary_uncapped` | 자본 계층 (보완자본 합계와 그중 50% 한도 밖 몫) |
| `liability_shift` | 누적 계리가정 변화 (손해율 +, 해지율 ×, 인플레 +, 10년 금리 +, 기울기 +, 무·저해지 해지율 ×) |
| `rate_level` | 기준 대비 누적 금리 변화 |
| `equity_scale`, `equity_units` | 남은 주식 노출(가격 포함), 남은 주식 수량(매도만 반영) |
| `retention`, `inforce` | 출재 후 보유율, 해지 후 남은 보유계약 비율 |
| `sub_debt_issued` | 누적 후순위채 발행액 (조원) |
| `rate_floored`, `outside_calibration` | 금리 하한에 걸렸는지, 보정 범위 밖 상태인지 |

경로 전체는 `simulate_path(model, xs, regimes, actions)`로 한 번에 돌릴 수 있다. 대응 비용 내역(출재·주식 매각·후순위채 캐리·포기한 배당, 조원)은 `model.action_cost(prev, x, action)`으로 얻는다.

## 유효 범위와 실행 가능 조건

- 보정 범위: 누적 금리 ±200bp, 손해율 누적 ±10%p, 해지율 배수 0.5~2, 무·저해지 해지율 배수 0.4 이상. 이 밖에서는 `outside_calibration`이 참이 되고, 결과는 외삽이다(공시 민감도는 ±100bp까지만 있다).
- 금리 하한: 시장 금리 0.5%(기준 4.1%에서 누적 −3.6%p) 아래로의 하락은 반영하지 않는다.
- 일반 `BalanceSheet`를 넘기면 보유율·보유계약·누적 발행액 같은 상태가 기본값으로 돌아가 여러 해 한도를 우회할 수 있다. 최적화에서는 받은 `InsurerState`를 그대로 넘긴다.
- 기준 수준(2026-06): 국고 10년 4.09%, 장기선도금리 4.3%, 최종관찰만기 23년, 장기 손해율 85%(모델포인트), 첫해 해지율 7~10%.
- 실행 가능 조건 (`feasible_actions`, 최적화에서 경계가 생기는 곳):
  - 후순위채: 스프레드 편차 ≤ +150bp, 직전 비율 ≥ 100%, 보완자본 한도 여유 있음, 누적 발행 ≤ 기준 가용자본의 10%
  - 출재: 누적 보유율 ≥ 60%
  - 주식 축소: 남은 수량 ≥ 기준의 40% (A는 59%), 남은 노출 > 0
  - 배당 중단: 항상 가능. 비율 100% 미만이면 배당이 자동으로 멈춘다.
- 대응 한도 값(60%, 10%, 40%)은 잠정값이다. 연구 진행에 따라 바뀔 수 있다.

## 구조

| 파일 | 내용 |
|---|---|
| `kics.py` | 대분류 위험액 상관 합산, 법인세조정 (별표22) |
| `insurer.py` | 연 단위 이월(`StylizedInsurer`), 자본 계층, 경과조치, 예정 제도변화, 경영대응 |
| `liability.py` | 장기 건강보험 모델포인트 현금흐름, BEL, 장기 하위위험, 위험마진 |
| `fit_liability.py` | 모델포인트를 공시 하위위험·CSM에 맞춤 → `data/liability_fit.json` |
| `archetypes.py` | 공시 → 회사 A~D 모수. 공시로 못 정하는 값은 `ASSUMPTIONS` |
| `sensitivity.py` | `ASSUMPTIONS`를 하나씩 ×0.5/×1.5로 바꿔 대표 결과 변화 (`python -m model.sensitivity`) |
| `backtest.py` | 과거 시장 변화로 K-ICS 비율 기여를 계산해 IR 변동요인과 비교 |
| `report.py` | 보정표 `docs/모형_보정표.md` 생성 (`python -m model.report`) |
| `data/disclosures_2026h1.csv` | 사용한 공시 수치와 출처 URL (정기경영공시, IR, FISIS, 일부 뉴스) |
| `data/market_levels.csv`, `data/kics_walk.csv` | 재현 검증용 시장 수준(ECOS 등)과 삼성화재 IR 변동요인 |

## 위험요인이 들어가는 경로

| 위험요인 | 가용자본 | 요구자본 |
|---|---|---|
| 금리 수준 | 자산: 공시 순민감도 − 부채 몫. 부채: BEL 할인율 | 시장위험액 (공시 표 − 부채 장기위험 몫) |
| 금리 기울기 | 부채 BEL 할인율 (만기에 선형). 자산 노출 0 가정 | — |
| 최종관찰만기 일정 | 부채 곡선이 해마다 바뀜 (23→24년, 2028). LTFR 4.3% 수렴 | — |
| 신용스프레드 | 자산: FISIS 신용자산 × 회사별 듀레이션(대출 3.75년, 유가증권 6년 가중). 부채: 변동성조정 10% 전가 (금감원 VA 이력 회귀) | — |
| 주가 | 주식 노출 × 가격 변화율 (세후). 노출은 주가·매도에 따라 이월 | 주식위험액 탄력성 |
| 손해율·해지율·사업비 | 현금흐름 부채의 BEL·위험마진 재계산. 신계약 이익은 그 해 가정 악화분만큼 감소(1년 시차 재가격) | 장기손해보험위험액 재계산 |
| 체제 점프 | 손해율 개편: 손해율 +1.3~2.4%p, 해지율 개편: 무·저해지 해지율 ×0.5 (과거 개편의 CSM 영향에 맞춤) | 같은 경로 |

## 그 밖의 장치

- 자본 계층: 보완자본 50% 한도(해약환급금 부족분 재분류는 한도 밖), 기본자본비율 계산.
- 법인세조정: 기본요구자본 × 평균세율, 기준 시점의 1.2배 한도.
- 경영대응: 출재(보험위험 비례 감소, 손해율이 오르면 비용 증가), 주식 축소(스프레드가 오르면 매각 손실, A는 전략 지분 59% 하한), 후순위채(스프레드 +150bp 초과·비율 100% 미만·한도 소진 시 불가, 음의 캐리), 배당 중단. 비율 100% 미만이면 배당이 자동으로 멈춘다.

## 주의

- 모든 모수의 출처는 `InsurerParams.sources`에 있다. 공시가 없어 대용치를 쓴 항목은 `sources["gaps"]`에 회사별로 남는다.
