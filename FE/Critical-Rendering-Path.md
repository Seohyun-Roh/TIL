# Critical Rendering Path(중요 렌더링 경로)

### 브라우저가 하나의 화면을 그려내는 과정

![image](https://user-images.githubusercontent.com/76952602/141133329-2d1d9094-5471-4b52-8da0-18135c956e1f.png)

1. 서버에서 응답으로 받은 HTML 데이터를 파싱한다.
2. HTML을 파싱한 결과로 DOM Tree를 만든다.
3. 파싱하는 중 CSS 파일 링크 만나면 CSS 파일 요청해 받아온다.
4. CSS 파일 읽어 CSSOM(CSS Object Model)을 만든다.
5. DOM Tree와 CSSOM이 만들어지면 이 둘을 사용해 Render Tree 만든다.
6. Render Tree에 있는 각각의 노드들이 화면에 어디에 어떻게 위치할 지를 계산하는 Layout 과정
7. 화면에 실제 픽셀을 Paint

---

## 1. 서버에서 응답으로 받은 HTML 데이터 파싱

- 브라우저 주소창에 url을 입력하면 브라우저는 해당 서버에 요청을 보낸다. 요청에 대한 응답으로 HTML 문서를 받아오게 되고, 이를 하나하나 파싱(Parsing)하면서 브라우저가 데이터를 화면에 그리는 과정이 시작된다.
- 미디어 파일을 만나면 추가로 요청을 보내 받아온다.
- Javascript 파일을 만나면 해당 파일을 받아와 실행할 때까지 파싱이 멈춘다.

## 2. HTML에서 DOM Tree로

![image](https://user-images.githubusercontent.com/76952602/141133810-3b02eb55-f604-4b08-adc8-313dc7ea8011.png)

- 브라우저는 읽어들인 HTML 바이트 데이터를 해당 파일에 지정된 인코딩(ex. UTF-8)에 따라 문자열로 바꾼다.
- 바꾼 문자열을 다시 읽어 HTML 표준에 따라 토큰으로 변환한다. 이 과정에서 `<html>`은 StartTag: html로, `</html>`은 EndTag: html로 변환된다.
- 이 토큰들을 다시 노드로 바꾸는 과정을 거친다. StartTag: html이 들어왔으면 html 노드를 만들고 EndTag: html을 만날 때까지 들어오는 토큰들을 html노드의 자식 노드로 넣는다. 과정이 끝나면 Tree 모양의 DOM(Document Object Model)이 완성된다.

## 3 & 4. CSS에서 CSSOM으로

- HTML을 파싱하다가 CSS링크를 만나면 CSS 파일을 요청해 받아온다. 받아온 CSS 파일은 HTML 파싱과 유사한 과정을 거쳐 Tree 형태의 CSSOM으로 만들어진다.
- CSS 파싱은 CSS 특성상 자식 노드들이 부모 노드의 특성을 계속해서 이어받는(Cascading) 규칙이 추가된다는 것을 제외하고는 HTML 파싱과 동일하게 이루어진다.
- CSSOM을 구성하는 것이 끝나야 비로소 이후의 Rendering 과정을 시작할 수 있다. (따라서 CSS는 Rendering의 blocking 요소라고 한다.)

## 5. DOM(Content) + CSSOM(Style) = Render Tree

- DOM과 CSSOM을 합쳐 Render Tree를 만든다. 이는 DOM Tree에 있는 것 중, 화면에 실제로 '보이는' 노드들로 이루어진다.
- 만약 CSS에서 `display:none`으로 설정했다면, 그 노드와 그 자식 노드 전부는 Render Tree에 추가되지 않는다. 마찬가지로 화면에 보이지 않는 `<head>` 태그의 내용도 Render Tree에 추가되지 않는다.

![image](https://user-images.githubusercontent.com/76952602/141135301-6ad4368f-f4db-4a32-8afa-a599ccc9f810.png)

- `<head>`태그와 display 속성이 none인 `<p>`태그 하위의 `<span>`태그가 사라진 것을 확인할 수 있다.
