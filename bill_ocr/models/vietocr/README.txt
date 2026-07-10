VietOCR text recognition — pretrained only (MC_OCR step 4).

Architecture: vgg_transformer (VGG19-bn + Transformer), same as VietOCR default config.
Source: https://github.com/pbcquoc/vietocr

Expected weight file in this folder:
  vgg_transformer.pth

Download (if missing):
  curl -L -o vgg_transformer.pth https://vocr.vn/data/vietocr/vgg_transformer.pth

Alternate mirror (Google Drive, MC_OCR vgg-transformer.yml):
  https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA

Runtime config: OCR/src/receipt_ocr/pipeline.py loads Cfg.load_config_from_name("vgg_transformer").
No fine-tune / retrain in production — only PICK KIE is retrained for new bills.

Legacy fine-tuned file vietocr_receipt.pth is no longer used.
