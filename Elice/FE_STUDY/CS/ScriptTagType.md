# script vs script async vs script defer

자세한 내용 정리: [script 태그는 어디에 위치해야 할까?](https://doooodle932.tistory.com/24?category=1007524)

- `<script>`: HTML 파싱이 중단되고 즉시 스크립트가 로드되며, 로드된 스크립트가 실행되고 파싱이 재개된다.
- `<script async>`: HTML 파싱과 병렬적으로 로드되고 스크립트를 실행할 때는 파싱이 중단된다.
- `<script defer>`: HTML 파싱과 병렬적으로 로드되고 파싱이 끝나고 스크립트를 로드한다. 보통 `<body>` 태그 직전에 스크립트를 삽입하는 것과 동작은 같지만 브라우저 호환성에서 다를 수 있으므로 그냥 body 태그 직전에 삽입하는 것이 좋다.
