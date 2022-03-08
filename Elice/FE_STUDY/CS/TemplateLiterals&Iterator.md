# 템플릿 리터럴, 반복자

스터디 학습 기록에 제출하기 위한 요약 내용입니다. 자세한 내용은 [링크](https://doooodle932.tistory.com/71)에서 확인해주세요.

## 템플릿 리터럴(Template literals)

- 복잡한 문자열을 쉽게 만들어주는 장치. 백틱(\`)으로 감싸고 변수명을 쓸 때는 `${변수명}`과 같은 형태로 씀.
- `태그드 템플릿 리터럴`: 템플릿 리터럴의 표현식을 분해할 수 있음.

```js
function taggedTemplateLiterals(str, ...rest) {
  console.log(str);
  console.log(rest);

  return 0;
}

let value1 = 10;
let value2 = 'ten';
let value3 = false;

const result = taggedTemplateLiterals`ABC${value1}EFG${value2}HIJ${value3}`;

console.log(result);

// ['ABC', 'EFG', 'HIJ', '', raw: Array(4)]
// [10, 'ten', false]
// 0
```

- 첫 번째 인자로 들어가는 str은 마지막에 값이 없더라도 빈 문자열이 들어가는 것에 주의.

## 반복자(Iterator)

- 반복을 위해 설계된 특정 인터페이스가 있는 객체.
- 두 개의 속성 `{value, done}`을 반환, next 메서드 가짐.
- next() 메서드를 반복적으로 호출해 명시적으로 반복 가능. value 프로퍼티-> 다음 시퀀스의 값 반환. done-> 시퀀스 마지막 값이 산출됐는지 여부를 boolean으로 반환.
- 이터러블한 객체(string, array, map, set)의 프로토타입 객체에는 모두 Symbol.iterator 메서드 있음.

```js
const array = [1, 2, 3];

// Symbol.iterator 메소드는 이터레이터를 반환한다.
const iterator = array[Symbol.iterator]();
console.log(iterator);

// 이터레이터의 next 메소드를 호출하면 value, done 프로퍼티를 갖는 이터레이터 리절트 객체를 반환한다.
console.log(iterator.next()); // {value: 1, done: false}
console.log(iterator.next()); // {value: 2, done: false}
console.log(iterator.next()); // {value: 3, done: false}
console.log(iterator.next()); // {value: undefined, done: true}
```

- `for ... of` 문은 내부적으로 이터레이터의 next 메서드 호출해 이터러블 순회. next 메서드가 반환한 이터레이터 리절트 객체의 value 프로퍼티 값을 for ... of 문의 변수에 할당. done 프로퍼티 값이 true이면 순회 중단.
