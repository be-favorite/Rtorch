# Rtorch 튜토리얼: The R's native deep learning library <a href='https://torch.mlverse.org'><img src='figures/torch.png' align="right" height="139" /></a>
R의 native한 딥러닝 라이브러리 {torch} 튜토리얼

* torch 튜토리얼: Still in progress :sweat:
* 참고자료: https://statisticsplaybook.github.io/deeplearning-playbook/
* Official docs: https://torch.mlverse.org/docs/index.html
* 딥러닝에 전반적 소개: https://be-favorite.tistory.com/8?category=897337

## Introduction
2020년 9월 Rstudio에서 운영하고 있는 블로그에서 R을 위한 딥러닝 라이브러리 {torch}를 소개하는 [글](https://blog.rstudio.com/2020/09/29/torch/)이 포스팅됐습니다. 저도 당시엔 몰랐으나, 몇달 전 유튜브 채널 [<슬기로운 통계생활>](https://www.youtube.com/c/statisticsplaybook/videos)에서 접할 수 있었죠..:sweat: 해당 패키지가 등장하기 전까지 R에서 구동하는 딥러닝 라이브러리들은 Python에 기반한 라이브러리(e.g. tensorflow)를 그저 R로 불러와 가져다 쓰는게 전부였습니다. 사실 딥러닝에 대한 모티베이션, 이론 등에 대한 관심을 갖고 잠깐 공부한 적이 있었으나 정작 실제 적용을 해 본 경험이 없었는데, Rtorch가 나왔다는 소식을 슬통 유튜브에서 접하고 꼭 활용해봐야 겠다는 결심을 했습니다. 그리고, 이 결심을 한 와중에 마침 슬기로운 통계쌩활님이 [<딥러닝 공략집 with Rtorch>](https://statisticsplaybook.github.io/deeplearning-playbook/)을 {bookdown}으로 제작하고 계셔서 공부에 불을 지필 수 있게 됐네요. 해당 레포에 정리해서 올릴 Rtorch 튜토리얼도 해당 공략집을 기반으로 제가 궁금한 것들을 이것저것 찾아보며 내용을 정리해서 올릴 예정입니다.😃 torch 뿐만 아니라 여러 딥러닝 알고리즘에 대한 공부도 함께 병행해야 할 것 같네요.

이 Repo가 {torch}와 딥러닝을 공부하는 R 유저들에게 조금이나마 도움이 됐으면 좋겠습니다. :blush:

<br>

<div align=center>
 
[![Github Badge](http://img.shields.io/badge/-Github%20profile-black?style=flat-square&logo=github&link=https://github.com/be-favorite)](https://github.com/be-favorite) 
[![Tistory badge](https://img.shields.io/badge/-Tistory%20blog-yellow?style=flat-square&logo=Blogger&link=https://be-favorite.tistory.com/)](https://be-favorite.tistory.com/) 
[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/taemo-bang-8b9999184/)](https://www.linkedin.com/in/taemo-bang-8b9999184/) 
[![Instagram Badge](https://img.shields.io/badge/-Instagram-dd2a7b?style=flat-square&logo=instagram&logoColor=white&link=https://www.instagram.com/qkdxoah/)](https://www.instagram.com/qkdxoah/) 

</div>
