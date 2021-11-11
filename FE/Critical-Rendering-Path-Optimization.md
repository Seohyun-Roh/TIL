# Critical Rendering Path 최적화

Critical Rendering Path 최적화란 HTML, CSS 및 자바스크립트 간의 종속성을 이해하고 최적화하는 것을 말한다.

## CSS

CSS는 렌더링 차단 리소스(Rendering Blocking Resource)이다.  
CSSOM이 생성될 때까지 브라우저는 렌더링하지 않는다.  
렌더 트리를 이용해 레이아웃과 페인팅 동작을 하므로, 렌더 트리를 만들 때 사용되는 HTML과 CSS는 둘 다 렌더링 차단 리소스이다.

최초 렌더링에 걸리는 시간을 최적화하려면 CSS를 간단하게 만들고 클라이언트에 최대한 빠르게 다운로드되어야 한다.

### 미디어 유형, 미디어 쿼리 이용

페이지가 인쇄될 때나, 대형 모니터에 출력하는 등 몇 가지 특수한 경우에만 사용되는 CSS가 있다면, 해당 CSS가 렌더링을 차단하지 않는 것이 좋다.  
이 경우, 미디어 유형과 미디어 쿼리를 사용하면 CSS 리소스를 렌더링 '비차단' 리소스로 표시할 수 있다.

```html
<link href="style.css" rel="stylesheet" />
<link href="style.css" rel="stylesheet" media="all" />
<link href="print.css" rel="stylesheet" media="print" />
<link href="portrait.css" rel="stylesheet" media="orientation:landscape" />
<link href="other.css" rel="stylesheet" media="min-width: 40em" />
```

- 첫 번째 stylesheet 선언은 미디어 유형이나 미디어 쿼리를 제공하지 않았기 때문에 모든 경우에 적용된다. 즉, 항상 렌더링을 차단한다.

- 두 번째는 미디어 유형을 `all`로 설정했다. 두 번째는 사실상 첫 번째와 똑같아서 이 또한 항상 렌더링을 차단한다.

- 세 번째는 미디어 유형을 사용한다. 컨텐츠가 print될 때만 적용되어 처음 로드될 때 페이지 렌더링이 차단되지 않는다.

- 네 번째는 미디어 쿼리를 `orientation:landscape`로 설정했다. 이는 기기의 방향이 가로일 때 렌더링을 차단한다.

- 다섯 번째는 미디어 쿼리를 `min-width: 40em`으로 설정했다. 이는 기기의 너비 조건이 일치하면 렌더링을 차단한다.

---

## Javascript

자바스크립트는 파서 차단 리소스(Parser Blocking Resource)이다.  
자바스크립트를 사용하면 컨텐츠, 스타일 등 거의 모든 것을 수정할 수 있기 때문에 자바스크립트 실행은 DOM 생성을 차단하고 페이지 렌더링을 지연시키게 된다.

### Javascript와 HTML의 종속성

자바스크립트는 DOM 노드와 CSS 스타일을 수정할 수 있는 강력한 기능과 유연성을 보여준다.

```css
/* style.css */
body {
  font-size: 16px;
}
p span {
  display: none;
}
```

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <link href="style.css" rel="stylesheet" />
    <title>Critical Path: Script</title>
  </head>
  <body>
    <p>Hello <span>web performance</span> students!</p>
    <script>
      var span = document.getElementsByTagName("span")[0];
      span.textContent = "interactive"; // change DOM text content
      span.style.display = "inline"; // change CSSOM property
    </script>
  </body>
</html>
```

자바스크립트를 사용하면 DOM에 접근해 화면에 표시되지 않는(display: none) span 노드를 가져올 수 있다.  
숨겨진 노드는 렌더 트리에는 표시되지 않지만 DOM에는 존재한다. (위의 예제는 `Hello interactive students!`가 표시됨)

자바스크립트를 사용하면 DOM에 새로운 노드를 추가, 제거, 수정할 수 있다.  
만약 위의 인라인 script가 span태그 위로 이동하면 span 노드를 찾을 수 없다는 에러가 발생한다. (getElementById('span')는 null 반환)  
이는 자바스크립트가 문서에 삽입된 정확한 위치에서 실행된다는 것을 보여준다.

HTML 파서는 script 태그를 만나면 DOM 생성 프로세스를 중지하고 자바스크립트 엔진에 권한을 넘긴다.  
자바스크립트 엔진의 실행이 완료된 후, 브라우저가 중지했던 시점부터 DOM 생성을 다시 시작한다.

따라서 JS는 화면에 그려지는 모든 태그들이 모두 파싱된 후인 body 태그를 닫기 직전에 script 태그를 선언하는 것이 좋다.

### 비동기 Javascript

기본적으로, 자바스크립트 실행은 파싱을 중지시킨다.  
위에서 살펴본 인라인 스크립트 뿐만 아니라 `<script src="app.js"></script>`와 같이 script 태그를 통해 포함된 자바스크립트 역시 파싱을 중지시킨다.

차이점이 있다면, script 태그를 이용해 자바스크립트를 실행할 경우 서버에서 자바스크립트를 가져올 때까지 기다려야 한다. 이로 인해 수십~수 천 밀리초의 지연이 추가로 발생할 수 있다.

스크립트가 페이지에서 무엇을 실행할 지 모르기 때문에 브라우저는 최악의 시나리오를 가정하고 파서를 차단한다.  
브라우저에 자바스크립트를 바로 실행할 필요가 없음을 알려준다면, 브라우저는 계속해서 DOM을 생성할 수 있고 DOM 생성이 끝난 후에 자바스크립트를 실행할 수 있게 된다.

이 때 사용할 수 있는 것이 `비동기 자바스크립트`이다.

```html
<script src="app.js" async></script>
```

이는 단순히 script 태그에 async 속성을 추가해주면 된다. 이는 자바스크립트가 사용 가능해질 때까지 브라우저에게 DOM 생성을 중지하지 말라고 지시하는 것이다.

## 리소스 우선순위 지정

브라우저에 전송되는 모든 리소스는 똑같이 중요한 것은 아니다.  
브라우저는 가장 중요한 리소스(스크립트나 이미지보다 CSS 우선)를 우선 로드하기 위해 가장 중요하다 생각되는 리소스를 추측해 먼저 로드한다.

### preload 속성

현재 페이지에서 빠르게 가져와야 하는 리소스에 사용되는 속성이다.

```html
<link rel="preload" as="script" href="super-important.js" />
<link rel="preload" as="style" href="critical.css" />
```

as 속성을 이용해 리소스의 유형을 알려줘야 하며, 브라우저는 올바른 유형이 설정되어 있지 않으면 미리 가져온 리소스를 사용하지 않는다.

`<link rel="preload" as="...">`는 브라우저가 반드시 리소스를 가져오게 만든다.  
리소스를 두 번 가져오게 하거나, 필요하지 않은 것을 가져오지 않도록 주의해야 한다.

preload를 통해 리소스를 가져왔지만 현재 페이지에서 3초 내로 사용되지 않는 리소스는 경고 메시지가 출력된다.

### prefetch 속성

미래에 사용할 수 있는 리소스를 가져와야 할 때 사용되는 속성이다.  
`<link rel="prefetch">`는 현재 페이지 로딩이 마치고 사용 가능한 대역폭(bandwidth)이 있을 때(다운로드 할 여유가 생겼을 때) 가장 낮은 우선순위로 리소스를 가져온다.

prefetch는 사용자가 다음에 할 행동을 미리 준비한다. 만약 현재 페이지가 1페이지라면

```html
<link rel="prefetch" href="page-2.html" />
```

위와 같이 사용해 2페이지를 먼저 가져와 준비한다. 이 때 주의할 점은 위와 같이 사용했더라고 page-2.html만 가져왔지 그에 필요한 리소스는 가져오지 않는다는 것이다.

> 참고: https://beomy.github.io/tech/browser/critical-rendering-path/
