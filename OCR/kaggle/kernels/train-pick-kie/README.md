# train-pick-kie

Train PICK KIE (MC-OCR step 5) on Kaggle GPU.

## Add Data (bắt buộc)

- `domixi1989/vietnamese-receipts-mc-ocr-2021` — ảnh + GT CSV
- `mainhatkhangb2205881/pick-train-code` — code train PICK (`OCR/vendor/pick`)

## Notebook = toàn cell, không file ẩn

| Cell | Nội dung |
|------|----------|
| Bootstrap | import + `WORK` |
| deps | allennlp shim + pip |
| dataset | tìm MC-OCR paths |
| build_data | CSV → TSV PICK |
| pick_setup | copy PICK từ dataset → `/kaggle/working/pick_train` |
| train / export | config, train, `model_best.pth` |

Không còn `write_text(pick_kaggle_common…)` hay `vendor_pick.zip`.

## Chuẩn bị dataset PICK code

```bash
python OCR/kaggle/kernels/sync_retrain_kernels.py
kaggle datasets version -p expense-ocr-nlu/OCR/kaggle/datasets/pick-train-code
```

## Push kernel

```bash
kaggle kernels push -p expense-ocr-nlu/OCR/kaggle/kernels/train-pick-kie
```
