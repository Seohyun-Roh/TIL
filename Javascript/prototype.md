# Javascript Prototype

## 1. 프로토타입 객체

- 자바스크립트 -> 프로토타입 기반 객체지향 프로그래밍 언어. 클래스 없이(Class-less)도 객체 생성 가능(ES6에서 클래스 추가됨)

### 프로토 타입이란?

- JS의 모든 객체 -> 자신의 부모 역할을 담당하는 객체와 연결되어 있음.
- 부모 객체의 프로퍼티 또는 메소드를 상속받아 사용할 수 있는데, 이런 부모 객체를 Prototype(프로토타입) 객체, Prototype이라 함.
- 프로토타입 객체는 생성자 함수에 의해 생성된 각각의 객체에 공유 프로퍼티를 제공하기 위해 사용

```javascript
var student = {
  name: "Lee",
  score: 90
};

console.log(student.hasOwnProperty("name")); //true

console.dir(student);
```

JS의 모든 객체는 [[Prototype]]이라는 인터널 슬롯(internal slot)을 가짐. 이 값은 null또는 객체. 상속을 구현하는데 사용됨.

[[Prototype]] 객체의 데이터 프로퍼티는 get 액세스를 위해 상속되어 자식 객체의 프로퍼티처럼 사용 가능. (set 액세스는 허용X.)

---

## 2. [[Prototype]] vs prototype 프로퍼티

모든 객체 -> 자신의 프로토타입 객체를 가리키는 [[Prototype]] 인터널 슬롯(internal slot)을 가지며, 상속을 위해 사용  
함수도 객체이므로 [[Prototype]] 인터널 슬롯을 가짐. 그러나 함수 객체는 일반 객체와는 달리 prototype 프로퍼티도 소유

### [[Prototype]]

- 함수를 포함한 모든 객체가 가지고 있는 인터널 슬롯
- 객체의 입장에서 자신의 부모 역할을 하는 프로토타입 객체를 가리키며, 함수 객체의 경우 Function.prototype을 가리킴

### prototype 프로퍼티

- 함수 객체만 가지고 있는 프로퍼티
- 함수 객체가 생성자로 사용될 때 이 함수를 통해 생성될 객체의 부모 역할을 하는 객체(프로토타입 객체)를 가리킴

---

## 3. constructor 프로퍼티

프로토타입 객체는 constructor 프로퍼티를 가짐. 이는 객체의 입장에서 자신을 생성한 객체를 가리킴

```javascript
function Person(name) {
  this.name = name;
}

var foo = new Person("Lee");

// Person() 생성자 함수에 의해 생성된 객체를 생성한 객체는 Person() 생성자 함수이다.
console.log(Person.prototype.constructor === Person);

// foo 객체를 생성한 객체는 Person() 생성자 함수이다.
console.log(foo.constructor === Person);

// Person() 생성자 함수를 생성한 객체는 Function() 생성자 함수이다.
console.log(Person.constructor === Function);
```
