# 실행 컨텍스트(Execution Context)

스터디 학습 기록에 제출하기 위한 요약 내용입니다. 자세한 내용은 [링크](https://doooodle932.tistory.com/68)에서 확인해주세요.

- 코드의 실행 환경에 대한 여러 정보를 담고 있는 개념. JS 엔진이 스크립트 파일을 실행하기 전 글로벌 실행 컨텍스트가 생성되고, 함수를 호출할 때마다 함수 실행 컨텍스트가 생성됨.
- 실행 컨텍스트가 생성되면 콜 스택에 쌓임.

## 실행 컨텍스트의 구성 요소

1. Lexical Environment
2. Variable Environment
3. this 바인딩

### Lexical Environment

- 변수 및 함수 등의 식별자 및 외부 참조에 관한 정보를 가짐. `Environment Record`, `outer 참조` 구성요소를 가짐.

### Variable Environment

- LE와 동일한 성격, var로 선언된 변수만 저장.

### this 바인딩

- 글로벌 실행 컨텍스트: strict mode라면 undefined로, 아니라면 글로벌 객체로 바인딩.(브라우저-> window, 노드-> global)
- 함수 실행 컨텍스트: 함수가 어떻게 호출되었는지에 따라 바인딩.

## 실행 컨텍스트의 과정

1. 생성 단계(LE 생성, VE 생성, this 바인딩): var은 undefined로 초기화. let, const는 아무 값도 안 가짐.
2. 실행 단계: 코드 실행하며 변수에 값 매핑.
