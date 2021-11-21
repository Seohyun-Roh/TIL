# CSS(Cascading Style Sheets) Syntax

CSS는 HTML의 각 요소의 style을 정의하여 화면에 어떻게 렌더링하면 되는지 브라우저에 설명하기 위한 언어이다.

HTML5 이전 버전의 HTML에는 style을 컨트롤할 수 있는 태그(font, center)가 존재해 CSS가 없이도 어느 정도 스타일 표현이 가능했으나 복잡하고 혼란스러운 언어가 되어버렸다.  
HTML5에서는 HTML은 정보와 구조화, CSS는 styling의 정의라는 본연의 임무에 충실한 명확한 구분이 이루어졌다.

---

## 1. Selector(선택자)

HTML 요소의 style을 정의하기 위해 스타일을 적용하고자 하는 HTML 요소를 선택할 수 있어야 한다.  
셀렉터는 스타일을 적용하고자 하는 HTML 요소를 선택하기 위해 CSS에서 제공하는 수단이다.

```css
h1 {
  color: red;
}
```

위와 같은 구문을 Rule Set(또는 Rule)이라 하며 셀렉터에 의해 선택된 특정 HTML 요소를 어떻게 렌더링할 것인지 브라우저에 지시하는 역할을 한다.  
Rule Set의 집합을 Style Sheet라고 한다.

## 2. HTML과 CSS의 연동

HTML은 CSS를 포함할 수 있는데, CSS를 가지고 있지 않은 HTML은 브라우저에서 기본으로 적용하는 CSS(user agent stylesheet)에 의해 렌더링된다.

### 2.1 Link style

HTML에서 외부에 있는 CSS 파일을 로드하는 방식. 가장 일반적으로 사용된다.

```html
<head>
  <link rel="stylesheet" href="css/style.css" />
</head>
```

### 2.2 Embedding style

HTML 내부에 CSS를 포함시키는 방식.  
HTML과 CSS는 서로 역할이 다르므로 Embedding style보다 Link style을 사용해 다른 파일로 구분되어 작성하고 관리되는 것이 바람직하다.

```html
<head>
  <style>
    h1 {
      color: red;
    }
  </style>
</head>
```

### 2.3 Inline style

HTML요소의 style 프로퍼티에 CSS를 기술하는 방식.  
Javascript가 동적으로 CSS를 생성할 때 사용하는 경우가 있다.  
하지만 일반적인 경우 Link style을 사용하는 편이 좋다.

```html
<body>
  <h1 style="color: red">Hello World</h1>
</body>
```

### 3. Reset CSS 사용하기

모든 웹 브라우저는 디폴트 스타일(브라우저가 내장하고 있는 기본 스타일)을 가지고 있어 CSS가 없어도 작동한다.  
그런데 웹 브라우저에 따라 디폴트 스타일이 상이하고, 지원하는 tag나 style도 제각각이어서 주의가 필요하다.

Reset CSS는 기본적인 HTML 요소의 CSS를 초기화하는 용도로 사용한다.  
즉 브라우저 별로 제각각인 디폴트 스타일을 하나의 스타일로 통일시켜 주는 역할을 한다.

자주 사용되는 Reset CSS

> Eric Meyer's reset
> https://meyerweb.com/eric/tools/css/reset/

> normalize.css
> https://necolas.github.io/normalize.css/
