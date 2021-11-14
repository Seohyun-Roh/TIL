# DOM(The Document Object Model)

DOM(문서 객체 모델)은 HTML, XML 문서의 프로그래밍 interface이다.  
DOM은 웹 페이지의 객체 지향 표현이며, 자바스크립트와 같은 스크립팅 언어를 이용해 DOM을 수정할 수 있다.

## DOM과 자바스크립트

DOM은 프로그래밍 언어는 아니지만 DOM이 없다면 자바스크립트 언어는 웹 페이지 또는 XML 페이지 및 요소들과 관련된 모델이나 개념들에 대한 정보를 갖지 못하게 된다.  
문서의 모든 element는 문서를 위한 DOM의 한 부분이다.

페이지 콘텐츠(the Page Content)는 DOM에 저장되고 자바스크립트를 통해 접근하거나 조작할 수 있다.

    API(web or XML page) = DOM + JS(scripting language)

### DOM에 어떻게 접근할 수 있는가?

DOM을 사용하기 위해 특별히 해야할 일은 없다.  
스크립트를 작성할 때, 문서 자체를 조작하거나 문서의 children을 얻기 위해 document 또는 window elements를 위한 API를 즉시 사용할 수 있다.

아래의 자바스크립트는 문서가 로그될 때(모든 DOM을 사용할 수 있기 되는 때) 실행되는 함수이다.  
이 함수는 새로운 h1 element를 생성하고, element에 text를 추가하며, h1을 이 문서의 트리에 추가한다.

```html
<html>
  <head>
    <script>
      // run this function when the document is loaded
      window.onload = function () {
        // create a couple of elements in an otherwise empty HTML page
        var heading = document.createElement("h1");
        var heading_text = document.createTextNode("Big Head!");
        heading.appendChild(heading_text);
        document.body.appendChild(heading);
      };
    </script>
  </head>
  <body></body>
</html>
```

## DOM의 핵심 Interfaces

- document.getElementById
- document.getElementsByTagName
- document.createElement
- parentNode.appendChild
- element.innerHTML
- element.style
- element.setAttribute
- element.getAttribute
- element.addEventListener
- window.content
- window.onload
- window.dump
- window.sctollTo

> 참조: https://developer.mozilla.org/ko/docs/Web/API/Document_Object_Model/Introduction
