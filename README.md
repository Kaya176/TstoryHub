# CNN을 이용한 문장 분류 모델 구현하기

## Introduction
본 코드를 작동하기 위해서는 fasttext file과 dataset을 따로 다운로드 받아야 합니다.

- fasttext : https://fasttext.cc/docs/en/crawl-vectors.html
- dataset : https://github.com/e9t/nsmc

본 프로젝트는 네이버 영화 리뷰 데이터셋을 이용하여 감성분석을 진행하는 프로젝트 입니다.
모델은 1-D CNN을 이용하였으며, 직접 CNN 구조를 구현한 점이 특징입니다.  

코드와 관련된 자세한 설명은 [제 블로그 게시글](https://kaya-dev.tistory.com/6)을 참고해 주시기 바랍니다!
## Quick start
```bash
>>> python -m model.py
```
