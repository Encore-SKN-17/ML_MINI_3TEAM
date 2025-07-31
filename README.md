# ML_MINI_3TEAM

# **프로젝트 명 : 중고차 가격 예측 모델 개발** 🚗

## 🌟**팀명**

## DBalance
<br>

## 🌟 **팀원 소개**  

| 이름      | GitHub ID                           |
|-----------|------------------------------------|
| 🧑‍💻 최동현  | [@donghyun4957](https://github.com/donghyun4957) |
| 👩‍💻 김수현  | [@K-SH98](https://github.com/K-SH98)         |
| 👩‍💻 김세한  | [@kimsehan11](https://github.com/kimsehan11) |
| 👨‍💻 성기혁  | [@venus241004](https://github.com/venus241004) |
| 👨‍💻 임길진  | [@LGJ0405](https://github.com/LGJ0405) |

<br>


## 🛠️ **기술 스택**

| **분류**   | **기술/도구**                                                                                       |
| **분류**         | **기술/도구**                                                                            |
|------------------|------------------------------------------------------------------------------------------|
| **언어**         | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python)     |
| **라이브러리**   | ![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy)       ![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas)   ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib)  |
| **협업 툴**      | ![GitHub](https://img.shields.io/badge/github-121011?style=for-the-badge&logo=github)   ![Git](https://img.shields.io/badge/git-F05033?style=for-the-badge&logo=git)          |

<br>


## 1. 프로젝트 개요

### 1.1 프로젝트 주제 선정 배경

<img width="500" height="200" alt="Image" src="./figure/background.png" />

프로젝트 주제 선정 배경
중고차 시장은 해마다 규모가 커지고 있으며, 차량의 연식, 주행거리, 연료 타입, 브랜드, 모델 등 다양한 요인에 따라 가격이 크게 달라집니다. 그러나 일반 소비자 입장에서는 이러한 요인들이 가격에 어떤 영향을 미치는지 직관적으로 파악하기 어렵습니다. 따라서, 차량의 주요 특성을 바탕으로 가격을 예측하는 모델을 만든다면 구매자와 판매자 모두에게 유용한 도구가 될 수 있습니다.

이 프로젝트는 다음과 같은 동기에서 시작되었습니다:

데이터 기반 의사결정: 감에 의존한 중고차 가격 책정이 아닌, 데이터를 기반으로 한 합리적인 판단 도출

실생활과의 연관성: 실제 거래가 이뤄지는 중고차 시장과 연결된 주제로, 실용성과 적용 가능성이 높음

회귀 문제 접근: 다중 변수 회귀 모델 학습 및 성능 비교를 위한 이상적인 사례 제공

이를 통해 다양한 회귀 모델(랜덤포레스트, 그래디언트 부스팅 등)의 성능을 비교하고, 모델의 해석 가능성과 예측력을 높이는 방법을 실습할 수 있었습니다.

판매가격과 주행거리를 확인한 결과, 중고차 판매자가 게시한 가격은 평균 748만3000원 수준이나 실제 취득가액은 평균 2129만6000원으로 2.8배 비쌌다.
출처 - https://www.mk.co.kr/news/business/10203008

<br>

### 1.2 프로젝트 목적
중고차 시장의 투명성과 효율성을 높이기 위해, 다양한 차량 특성(연식, 주행거리, 연료, 브랜드 등) 데이터를 기반으로 중고차 가격을 예측하는 머신러닝 모델을 개발합니다. 이를 통해 합리적인 가격 산정과 소비자·판매자 모두에게 도움이 되는 인사이트를 제공합니다.

<br>


## 2. 데이터 선택 및 구조

### 1.1 데이터 출처
- KAGGLE Car price prediction(used cars) 데이터셋
(https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars?resource=download)

<br>

---

### 1.2. 데이터 구조

#### **분석 타겟 컬럼**
- `Selling_Price` : 중고차 거래 가격 (예측 대상)

#### **주요 변수**
- `Car_Name` : 모델명
- `Year` : 차량 연식
- `Driven_kms` : 주행거리
- `Fuel_Type` : 연료 종류 (가솔린, 디젤, CNG 등)
- `Transmission` : 변속기 종류 (수동, 자동)
- `Present_Price` : 중고차 구매가
- `Selling_type` : 판매자 유형 (딜러, 개인)
- `Owner` : 소유주 수

---

## 3. 데이터 기초 통계량

|       | price    | year     | mileage   | engine_size | brand   | fuel   |
|-------|----------|----------|-----------|-------------|---------|--------|
| count | 10000    | 10000    | 10000     | 10000       | 10000   | 10000  |
| mean  | 12,500,000| 2015     | 85,000    | 1,600       | 현대    | 가솔린 |
| std   | 4,200,000| 4.2      | 38,000    | 400         | 다양    | 다양   |
| min   | 2,000,000| 2000     | 5,000     | 800         | ...     | ...    |
| 25%   | 9,000,000| 2012     | 60,000    | 1,400       | ...     | ...    |
| 50%   | 12,000,000| 2016     | 80,000    | 1,600       | ...     | ...    |
| 75%   | 15,000,000| 2019     | 110,000   | 2,000       | ...     | ...    |
| max   | 40,000,000| 2023     | 300,000   | 5,000       | ...     | ...    |

---

## 4. 데이터 전처리 및 EDA (탐색적 데이터 분석)

- 이상치 제거, 파생변수 생성, 필요한 변수 추출(drop)

### 4.1 데이터 조회

수치형, 범주형 데이터 확인

<img width="300" height="200" alt="Image" src="./figure/data_info.png" />

데이터 결측치 조회

    - 데이터 결측치 처리:

        | 컬럼명         | 결측치 개수 |
        |---------------|:-----------:|
        | price         |      0      |
        | year          |      0      |
        | mileage       |      0      |
        | fuel          |      0      |
        | brand         |      0      |
        | model         |      0      |
        | transmission  |      0      |
        | engine_size   |      0      |
        | color         |      0      |

        > `df.isnull().sum()` 결과, 결측값이 없어 별도 처리는 하지 않음


수치형 데이터 이상치 조회

<img width="800" height="250" alt="Image" src="./figure/ourlier.png" />

- 주행거리 이상치와 구매가 이상치는 분포에서 극단적인 값을 보여 제거
- 극단적인 이상치는 차량의 특성을 대표하는 결정적인 요인이 아니며 회귀성능에 미칠 영향에 비해 중요한 요인이 아님

<br>
수치형, 범주형 변수 조회

<img width="500" height="150" alt="Image" src="./figure/num_cate.png" />

### 4-2. 그래프 시각화

#### 판매 가격에 따른 수량 막대그래프

<img width="800" height="250" alt="Image" src="./figure/sell_count.png" />

#### 수치형 변수들과 판매 가격 산점도

<img width="800" height="250" alt="Image" src="./figure/num_sell.png" />

- 차량 연식을 자동차 나이로 계산한 파생변수 생성

#### 연료 유형에 따른 판매 가격 분포

<img width="800" height="250" alt="Image" src="./figure/fuel_sell.png" />


#### 변속기 유형에 따른 판매 가격 분포

<img width="800" height="250" alt="Image" src="./figure/tran_sell.png" />


#### 판매 유형에 따른 판매 가격 분포

<img width="800" height="250" alt="Image" src="./figure/selt_sell.png" />


#### 차량 모델별 판매량

<img width="800" height="250" alt="Image" src="./figure/car.png" />

#### Top 10 판매 차량 모델별 가격 분포

<img width="800" height="250" alt="Image" src="./figure/car_sell.png" />

- Top 10 판매차량 모델 및 그 외 범주로 구성된 파생변수 생성

#### 수치형 변수들 간의 상관관계 히트맵

<img width="400" height="350" alt="Image" src="./figure/heat_map.png" />

#### 수치형 변수들 간의 강한 상관관계 히트맵(|r|>=0.1)

<img width="400" height="350" alt="Image" src="./figure/heat_map_strong.png" />

---

### 5. 머신러닝 파이프라인

#### - 학습에 사용된 변수
<div align="letf">
  <img src="./figure/columns.png" width="400">
</div>
- Target : `Selling_Price`

#### - 변수 전처리 과정
<div align="letf">
  <img src="./figure/scaling.png" width="400">
</div>
- 수치형 변수 : StandardScaler 적용  <br>
- 범주형 변수 : LabelEncoder 적용

#### - 테스트 모델 및 하이퍼파라미터 후보군
<div align="letf">
  <img src="./figure/hyper.png" width="400">
</div>
- Linear Regression : 선형 회귀 <br>
- Ridge Regression : 릿지 회귀 (L2 정규화) <br>
- Lasso Regression : 라쏘 회귀 (L1 정규화) <br>
- Decision Tree : 결정 트리 회귀 <br>
- Random Forest : 랜덤 포레스트 회귀 <br>
- Gradient Boosting : 그래디언트 부스팅 회귀 <br>
- SVR : 서포트 벡터 회귀 <br>
- KNN : K-최근접 이웃 회귀 <br>
- XGBoost : eXtreme Gradient Boosting 회귀 <br>

각 모델별로 다양한 하이퍼파라미터 후보군을 설정하여 성능을 비교함

#### - 교차 검증 결과
<div align="letf">
  <img src="./figure/kcv.png" width="400">
</div>
- scoring은 neg_mean_squared_error를 사용함


#### - 최종 변수 결과
<div align="letf">
  <img src="./figure/best_model.png" width="400">
</div>
- 최적의 변수 조합과 모델 선정, 교차검증 결과를 통해 최종 성능 평가								   
### 5. 최종 인사이트 및 활용 방안
- **연식과 중고차 가격 간의 상관관계**
  - 차량 연식이 최신일수록 중고차 가격이 높아짐 (상관계수: **0.72**)  
  - 최신 연식 차량은 시장에서 더 높은 가치를 인정받고 있음을 확인
- **주행거리와 중고차 가격 간의 상관관계**
  - 차량의 주행거리가 많을수록 중고차 가격은 하락 (상관계수: **-0.68**)  
  - 소비자들이 짧은 주행거리를 선호하는 경향 확인
- **브랜드별 중고차 시장 점유율**
  - 특정 브랜드가 전체 시장에서 큰 비중을 차지  
  - 브랜드 인지도 및 신뢰도가 시장 점유율에 영향을 미침
- **정책적 활용 방안**
  - 소비자와 판매자 모두에게 명확한 가격 산정 기준 제공  
  - 브랜드별 차량 관리 및 품질 인증 제도 도입  
  - 연식과 주행거리 등 주요 변수에 따른 가격 가이드라인 제공으로 시장 투명성 향상

---

### 6. 최종 인사이트 및 정책적 제언
| 변수명         | 설명             | 의미           |
|----------------|------------------|----------------|
| Present_Price  | 현재 판매가      | 시장가(시세)   |
| Selling_Price  | 실제 판매가      | 실거래가       |
| Driven_kms     | 주행 거리        | 차량 누적 주행 거리 (km) |
| Year           | 연식             | 차량 출시 연도 |
| Owner          | 이전 소유자 수   | 차량을 거친 소유자 수 |
### 양의 상관 관계
- Present_Price – Selling_Price : 0.88
	- Present_Price(시장가)가 높을 수록 Selling_Price(실거래가) 역시 높게 형성 됨.

- Year - Selling_Price : 0.24
	- Year(연식)이 최근일수록 Selling_Price(실거래가)가 높아지는 경향이 보이긴 함
	- 그러나 Car_Name_Grouped(차종), Driven_kms(주행거리) 등 다양한 변수가 많아 단일 변수로는 영향이 미미함.

- Driven_kms - Present_Price : 0.20
	- Driven_kms(주행거리)가 많을수록 Present_Price(시장가)가 높아지는 경향이 보이긴 함.
	- 그래프를 확인하면 대부분의 차량이 10만 km 이하에 몰려 있고, 이 구간에서도 판매가가 넓게 분포되어 있어 Year(연식), Car_Name_Grouped(차종) 등 여러 변수가 존재해 주행거리와 시장가 간엔 상관관계가 거의 없다고 보는게 타당함.

#### 정책적 시사점
   1. 투명한 가격 산정을 위한 가이드라인 제공  
   2. 차량 품질 인증 및 관리 시스템 구축으로 소비자 보호 및 시장 효율성 제고
   3. 데이터 기반 중고차 거래 플랫폼 활성화

---

### 7. 한계점

1. **데이터의 한계**

   - 데이터가 오래된 경우 연식 변수의 중요도가 시점에 따라 달라질 수 있음. 예를 들어, 2020년 데이터에서의 2018년식 차량과 2025년 데이터에서의 2018년식 차량은 시장에서 서로 다른 가치를 가질 수 있음. 

   - 데이터의 지역적 편중, 브랜드별 불균형 등도 존재하여 실제 시장 전체를 대표하기에는 한계가 있음.

2. **변수 선택의 한계**
   - 본 모델은 연식, 주행거리 등 주요 변수 위주로 분석을 진행하였으나, 실제 중고차 가격에는 소비자 선호도, 지역, 계절, 경제 상황, 유행 등 다양한 요인이 복합적으로 작용함. 이러한 변수들이 데이터에 포함되지 않아 예측 모델의 설명력과 현실 적용력이 제한됨.

   - 예를 들어, 특정 지역에서는 같은 연식·주행거리 차량이라도 가격 차이가 발생할 수 있고, 계절에 따라 SUV나 오픈카 등 특정 차종의 수요가 달라질 수 있음. 이러한 요인들이 반영되지 않아 모델의 예측 결과가 실제 시장과 다를 수 있음.

3. **실제 적용의 한계**
   - 본 모델은 실제 중고차 거래 플랫폼에 바로 적용하기 위해서는 추가적인 검증과 보완이 필요함. 예측 결과가 실제 거래 가격과 차이가 있을 수 있으며, 시장 데이터와의 연동, 사용자 피드백 반영, 지속적인 모델 업데이트 등에 대한 보완이 필요함.

4. **데이터셋 한계에 대한 근거**
    - 브랜드 평판, 온라인 리뷰가 구매자 선호에 결정적인 역할을 함을 보여주는 논문
    -  https://www.itm-conferences.org/articles/itmconf/pdf/2025/01/itmconf_dai2024_04033.pdf 

<br>

🌈 **팀원 한 줄 회고**
<table>
  <tr>
    <th style="width:80px;">이름</th>
    <th>회고 내용</th>
  </tr>
  <tr>
    <td>최동현</td>
    <td>회고 1.</td>
  </tr>
  <tr>
    <td>김세한</td>
    <td>협업 중 코드 버전 관리와 충돌 해결이 가장 어려웠음. Git의 브랜치 전략을 도입하고, 커밋 메시지 규칙을 정해 팀원 간 소통을 강화함으로써 효율적으로 협업할 수 있었음. 또한 다양한 머신러닝 모델을 적용하며 데이터 전처리와 하이퍼파라미터 튜닝의 중요성을 체감함.</td>
  </tr>
  <tr>
    <td>김수현</td>
    <td>주제 정하는 것부터 어느 하나 쉬운일이 없었지만, 팀원들과 함께 머리 맞대고 해결해 나가면서 끝까지 완성할 수 있었어요! EDA는 조금 아쉬웠지만, 짧은 시간 안에 머신러닝을 직접 경험해볼 수 있어서 정말 뜻깊은 시간이었습니다.</td>
  </tr>
  <tr>
    <td>성기혁</td>
    <td>처음엔 입력 데이터의 대부분이 범주형 데이터인 것을 깊이 고려하지 않고 ML 프로젝트를 진행했지만, 조언을 통해 머신러닝을 적용할 필요조차 없는 문제였다는 걸 인식하게 되었고, ‘중고차 가격 예측’으로 주제를 변경하며, 데이터 특성에 따라 적절한 방법론을 선택하는 안목이 필요하다는 걸 배웠다.</td>
  </tr>
  <tr>
    <td>임길진</td>
    <td>머신러닝 프로젝트를 하면서 주제 선정에 가장 어려움을 느꼇습니다. 무언가 주제를 선정해도 해당 주제가 머신러닝이 필요한게 맞나, 또는 머신러닝 영역에서 가능한가를 제대로 확인하지 않고 진행해 주제가 부적합한것을 나중에나 알게 되어 판을 갈아 엎는일이 발생해 충분한 데이터 수집이나 검증이 부족해 아쉬움이 남았습니다.</td>
  </tr>
</table>
