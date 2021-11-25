# Javascript의 Event Loop

JS를 공부하다 보면 '싱글스레드 기반으로 동작하는 자바스크립트', '이벤트 루프를 기반으로 하는 싱글 스레드 Node.js'등의 말을 종종 듣는다.  
정말 싱글 스레드인지, 싱글 스레드는 무엇이며 어떻게 싱글 스레드인지, 이벤트 루프는 무엇인지에 대해 알아보기 위해 자바스크립트가 동작하는 환경(Environment)과 자바스크립트를 해석하고 실행시키는 엔진에 대해 공부한다.

## Javascript Engine

#### Javascript Engine !=== Rendering Engine

JS를 해석하는 Javascript Engine과 웹 브라우저에 화면을 그리는 Rendering Engine은 다른 것이다.  
Rendering (또는 Layout) Engine은 HTML과 CSS로 작성된 마크업 관련 코드들을 웹 페이지에 'rendering'라는 역할을 한다.

Javascript Engine이란 JS로 작성한 코드를 해석하고 실행하는 **인터프리터**다.  
주로 웹 브라우저에서 이용되지만 최근에는 node.js가 등장하면서 server side에서는 V8과 같은 Engine이 이용된다.

대부분의 자바스크립트 엔진은 크게 다음의 세 영역으로 나뉜다.

`Call Stack`, `Task Queue(Event queue)`, `Heap`

그리고 추가적으로 `Event Loop`라는 것이 존재해 Task queue에 들어가는 task들을 관리한다.

![image](https://user-images.githubusercontent.com/76952602/143388284-49e0ba3f-d54d-4d1d-b18b-44b7e79b0b85.png)

### Call Stack

Javascript는 **단 하나의 호출 스택(Call stack)**을 사용한다.  
이런 특징 때문에 자바스크립트 함수가 실행되는 방식을 "Run to Completion"이라고 한다. 이는 하나의 함수가 실행되면 이 함수의 실행이 끝날 때까지 다른 어떤 Task도 수행될 수 없다는 의미이다.

요청이 들어올 때마다 해당 요청을 순차적으로 호출 스택에 담아 처리한다. 메소드가 실행될 때 Call Stack에 새로운 프레임이 생기고 push되고, 메소드의 실행이 끝나면 해당 프레임은 pop되는 원리이다.

```javascript
function foo(b) {
  var a = 10;
  return a + b;
}

function bar(x) {
  var y = 2;
  return foo(x + y);
}

console.log(bar(1));
```

위 코드에서 bar함수를 호출했으니 'bar'에 해당하는 스택 프레임이 형성되고 그 안에는 'y'와 같은 local variable과 arguments가 함께 생성된다.  
그리고 bar함수가 'foo'함수를 호출하고 있다. 아직 bar 함수가 종료되지 않았으니 pop하지 않고 호출된 'foo'함수가 Call Stack에 push된다. ('bar'함수 호출과정과 동일한 과정 거침.)

'foo'함수에서는 a+b라는 값을 return하면서 함수의 역할을 마쳤으므로 stack에서 pop된다.  
다시 'bar'함수로 돌아와서 'foo'함수로부터 받은 값을 return하면서 bar함수도 종료되고 stack에서 pop된다.

### Heap

동적으로 생성된 객체(인스턴스)는 힙(Heap)에 할당된다. 대부분 구조화되지 않는 '더미'같은 메모리 영역을 'heap'이라 표현한다.

### Task Queue(Event Queue)

자바스크립트 런타임 환경에서는 처리해야 하는 task들을 임시 저장하는 대기 큐가 존재한다.  
그 대기 큐를 Task Queue 또는 Event Queue라고 한다. 그리고 **Call Stack이 비어졌을 때** 먼저 대기열에 들어온 순서대로 수행된다.

```javascript
setTimeout(function () {
  console.log("first");
}, 0);
console.log("second");

// console >>
// second
// first
```

위의 코드에서 setTimeout에 0을 주었으니 first가 먼저 실행될 것 같지만, second가 먼저 출력된다.

자바스크립트에서 비동기로 호출되는 함수들은 Call Stack에 쌓이지 않고 Task Queue에 enqueue된다.  
자바스크립트에서는 이벤트에 의해 실행되는 함수(핸들러)들이 비동기로 실행된다. 자바스크립트 엔진이 아닌 Web API 영역에 따로 정의되어 있는 함수들을 비동기로 실행된다.  
( 즉 Web API 영역에 따로 정의되어 있는 함수들은 Call Stack에 쌓이지 않고 Task Queue에 enqueue되기 때문에 Call Stack이 비어졌을 때 들어온 순서대로 실행된다! )

```javascript
function test1() {
  console.log("test1");
  test2();
}

function test2() {
  let timer = setTimeout(function () {
    console.log("test2");
  }, 0);
  test3();
}

function test3() {
  console.log("test3");
}

test1();
```

위와 같은 코드를 실행하면 먼저 test1()이 실행되어 "test1"이 출력된 후 test2()가 호출된다. setTimeout 함수가 실행되고 콜스택에 들어간 다음 바로 빠져나온다. 그리고 내부에 걸려있던 핸들러는 call stack 영역이 아닌 event queue 영역으로 들어간다.  
그리고 test3 함수가 콜스택으로 들어간다.

"test3"이 출력되고 작업을 모두 마친 test3()가 콜스택에서 pop되고, 이어서 test2()와 test1()까지 콜스택에서 pop된다.  
이 때 이벤트 루프의 콜스택이 비어있게 되고, 바로 이 시점에 queue의 head에서 하나의 event를 가져와 Call Stack으로 넣는다.  
이 이벤트를 setTimeout 함수 내부에 있던 익명함수이다.

즉 test3가 끝나고, test2가 끝나고, test1이 끝나서 나서 이벤트 루프에 의해 하나의 event가 dequeue된 다음 콜스택으로 들어가 실행된다. 그러므로 이벤트에 걸려있는 핸들러는 절대 먼저 실행될 수 없다.

```javascript
// console >>
// test1
// test3
// test2
```

queue에 처리해야 할 이벤트(또는 태스크)가 존재하면 해당하는 이벤트를 처리하거나 작업을 수행한다. 그리고 다시 queue로 돌아와 새로운 이벤트가 존재하는지 파악하는 것이다.

즉 Event Queue에서 대기하고 있는 Event들은 **한 번에 하나씩** Call Stack으로 호출되어 처리된다.

> https://asfirstalways.tistory.com/362
