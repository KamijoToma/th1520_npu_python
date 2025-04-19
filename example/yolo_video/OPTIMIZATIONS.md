# yolo_video

## Optimize tries records of yolov5n_para.py

### Try1: add preprocess queue

Status: **Failed**

Before optimize:

```
Time spend: 98.33288764953613 = 7.667831363676594fps
Infq: 0.40716180371352784
Inputq: 3.9787798408488064
OutQ: 0.001326259946949602
```

After optimize:

```
Time spend: 103.10340571403503 = 7.313046497137787fps
Infq: 0.3103448275862069
Inputq: 3.9045092838196287
OutQ: 0.003978779840848806
PreQ: 7.936339522546419
```
