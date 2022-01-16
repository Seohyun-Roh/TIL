복사한 주소 터미널에 붙여넣기: `Shift + Ctrl + v`

### git branch

- 기능 단위로 독립된 작업할 때
- 메인 branch-> 안정적
- 토픽 branch-> 기능추가, 버그 수정 등 단위 작업
  `git branch [브랜치 이름]` -> 브랜치 생성
  `git checkout [브랜치 이름]` -> 브랜치 전환
  `git checkout [16진수 해시]` -> 해당 해시로 HEAD를 옮길 수 있음. 해당 파일 내용 확인 가능

### fast-foward

- merge: 다른 브랜치에서 작업 마치고 master로 통합할 때
- `git checkout master`로 마스터로 이동 -> `git merge like_feature`하면 마스터 브랜치를 중심으로 like_feature 브랜치가 병합됨. like_feature가 가지고 있는 내용은 master가 가지고 있는 내용과 같음. like_feature내용이 master 브랜치에서 업데이트된 내용이기 때문에 곧바로 merge가 됨. -> 이렇게 merge가 이루어지는 것을 `fast-forward`라고 함.

### 갈라지는 branch

- 파일을 동시에 수정하는 경우
- `git log --graph --all`을 사용하면 commit graph를 확인할 수 있음.

- 마스터로 이동한 후 merge like_feature하면 마스터에서 수정한 내용이 like feature의 내용을 받아들여 둘이 합쳐진 형태. 하지만 아까와는 달리 둘이 동시에 같은 체크포인트를 가르키지 않음.
- `git branch --merged` 명령어를 통해 merge된 브랜치 확인 가능.
- 토픽 브랜치는 각 기능의 개발이 끝나면 삭제해주는 것이 대부분임. 따라서 `git branch -d <branch name>`을 이용해 삭제 가능.

### merge conflict

- merge한 두 브랜치에서 같은 파일을 변경했을 때 발생. 충돌발생한 파일을 열어 최종 수정을 해준 후 깃에서 추가한 '<<<<', '====', '>>>>'가 포함된 행 삭제. 수정 완료 후 git add, commit을 한 후 다시 merge.
- 충돌 방지-> master branch의 변화를 지속적으로 가져와서 충돌이 발생하는 부분을 제거하기. (제일 좋은 방법은 마스터가 자주 변경되는 일이 없도록 하는 것! 배포 가능한 안정적인 버전이어야 하기 때문.)

---

### git 원격 저장소

- `git remote add origin [https://gitlab.com(웹 호스트 서비스)/group(그룹명)/project(프로젝트명)`
- `git remote` 명령어 -> 연결된 원격 저장소 확인.
- `git remote rename origin git_test` -> 원격 저장소 단축 이름을 origin에서 git_test로 변경.
- `git remote rm ~`으로 삭제도 가능.

### git clone

- `git clone` 명령어를 사용하면 현재 폴더 내에 새로운 폴더를 하나 더 만듬.
- 현재 폴더를 저장소로 쓰고 싶다면 git clone 명령어의 마지막에 `.`을 찍어주면 됨.

### 원격 저장소 동기화

- 원격 저장소에서 데이터 가져오기 + 병합 -> `pull`
- 원격 저장소에서 데이터 가져오기 -> `fetch`
- push -> 로컬에서 작업한 내용 원격 저장소에 반영. 이 때 다른 사람이 먼저 push한 내용이 있다면 pull이나 fetch를 이용해 먼저 merge해준 후 push해주기.

- pull이 이루어지지 않는 경우 -> 다른 사람이 올린 commit의 내용과 내 컴퓨터에 존재하는 내용이 서로 충돌할 때. 이런 현상은 하나의 브랜치에서 여러 사람이 동시에 작업 시 발생확률 높아짐. 따라서 여러 개의 브랜치를 나누고 각자 브랜치에서 작업 후 하나씩 합쳐가는 방식 이용 시 충돌 방지 가능.

- `git remote add origin ~링크` 이는 원격 저장소의 단축 이름을 origin으로 지정한다는 의미.(다른 이름으로도 가능)

- `git remote -v` -> 지정한 저장소의 이름과 주소 함께 보기 가능.
