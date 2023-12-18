Semantic Segmentation - based Split Merge

## Configuration
Configuration cho train/infer được load trực tiếp từ yaml file cùng tên.

## Report
- Hiện module tốt nhất là 0.05 vector + 0.95 segmentation (95.51 mIOU với 1024 và 92.0 với 860). Tuy nhiên do ảnh hưởng của vector nên không hoạt động tốt với những bảng nghiêng
- Có thể cải thiện bằng cách giảm trọng số của vector nhưng không cải thiện nhiều. Hợp lí nhất là collect thêm data.

## Training
- Sửa lại hoặc bỏ phần wandb đi trước khi train (xem file [train_split_nowdb.py](./tools/train_split%20nowdb.py))
- Chạy file bash:

```bash
bash bash_scripts/train_split_v3.sh 
```

## Infer 
- Sửa các thông số trong mục Test của config yaml file tương ứng (nhìn thấy trong phần main của file .py)
- Chạy file bash
```bash
python tools/infer_split.py
```