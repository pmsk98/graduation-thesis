#### 졸업논문
- 시가총액 100위 fdr_datareader 패키지로 크롤링
- 2008년 주가가 없는 경우는 제외 +  우선주 제외 (삼성전자우)

- talib 패키지에서 모멘텀 지표 선택
- 모멘텀 지표 파라미터 조정 -> 파라미터 조정 기준: 키움증권 참고
- trian/ test 검증 두 가지 방법으로 비교


1. walk-forward 교차 검증 방법
ex) train(2009 ~ 2015) / test(2016)  -> train(2009~2016) / test(2017)

2. blocking walk forward 교차 검증 방법
ex) train(2010 ~2016) / test(2017)  ->  train(2011~2017) / test(2018)

train set과 test set은 위의 방식으로 진행

###Model
1. logistic regression
2. decsion tree
3. naive bayes
4. randomforest
5. SVM
6. K-nn
7. neural_network(MLP classifier)
8. voting(hard voting)
7. gbm
- 별도의 파라미터 조정은 없고 default 값으로 진행


###Position label 생성
- 예측값을 기준으로 Position 지정
- ex)
- 2016-01-04 down 
- 2016-01-05 up -> buy
- 2016-01-06 up -> holding
- 2016-01-07 down -> sell

### ratio 작성
- 거래 횟수
- 승률
- Average gain
- Average loss
- Payoff ratio
- Profit factor




