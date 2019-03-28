# WeDoTrend
Hynix Trend Viewer (신입사원 python 교육 프로젝트)

* 실행 파일: ./code/web_flask.ipynb or ./code/web_flask.py

## **Member** : 김다혜, **박헌준(조장)**, 이수연, 채선율 

## 목적 및 의의
### Trend Chart, Top Keywords : 하이닉스 구성원의 복지 서비스에 대한 의견의 변화 및 실시간 관심사 확인
### 

## 1. 진행 상황
### 1-0. 첫 페이지 만들기
* Title : Hynix Trend Viewer
* ~카드 형식의 link : 각각의 카드가 1-1, 1-2 로 연결~

### 1-1. Trend Chart 구성
  * ~SK hystec 고객의 소리 크롤링~
  * Topic Modeling
  * Point들에 대한 세부 사항을 report

### 1-2. Weekly Top Keywords, Monthly Top Keywords.
 * 주간 인기 검색어, 월간 인기 검색어

### 1-3. 감정 분석을 통한 카테고리 별 긍·부정 정도 (만족도)
 * 감정 분석 시행
 * 이거 

### 1-3. Additional works (후보 - contents의 의미)
* VoC 유형이 채워친 글을 바탕으로 유형이 있지 않은 글을 분류 - 하이스텍 직원들이 편리하게(??) 무슨 카테고리인지 알 수 있음
* 최근 몇 개월 간의 데이터를 이용하여 카테고리 별로 직원들의 감정(긍-부정) 분석 - 해당 카테고리에 대한 만족 정도를 알 수 있음
                                                                          ->회사 내의 정책에 대한 만족도를 파악. 정책의 적절성
* 칭찬 일지도 모르는 ...
* 당신이 알 수도 있는...
** 답변글도 있음 -> 답변 글은 따로 빼서 다루어야 함 (중복적인 내용)
** 답변이 있는 글과 있지 않은 글의 경우로 나누기 -> 어떠한 의견에 답변이 달리는지 확인하기
** 관련 solution
6. 3d map. 


질문에서의 불만의 정도는 수치화 되어있고, 각각의 글을 vector화 시켜서 regression.


## 2. 고려 사항
* 1-1 : Topic Modeling에서 교통, 식당 등의 초기 카테고리로 생각하여 
* 1-3 감정 분석 : 카테고리 별로 진행
