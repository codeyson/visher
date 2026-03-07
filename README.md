# Visher
**Voice Phishing Detection using RawNet**

This project detects **AI-generated (fake) voices vs real voices** using a RawNet-based model trained on the **LibriSeVoc dataset**.

---

## Requirements

- Python **3.10.11**

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

#### Quick Training
```bash
python main.py --data_path ./LibriSeVoc  --batch_size 4  --num_epochs 2  --model_save_path ./checkpoints
```
#### Full Training
```bash
python main.py --data_path ./LibriSeVoc  --batch_size 64 --num_epochs 50  --lr 0.0001 --model_save_path ./checkpoints
```

## Evaluation
#### For Real Voice
```bash
python eval.py --input_path ./LibriSeVoc/gt/250_142286_000031_000006.wav --model_path ./checkpoints/best_model.pth
```

#### For AI Generated Voice
```bash
python eval.py --input_path ./LibriSeVoc/wavernn/696_92939_000008_000001_gen.wav --model_path ./checkpoints/best_model.pth
```

## Model Optimization
Run in Sequence:
```bash
python pruning.py
python quantization.py
python quantized_inference.py --input_path ./LibriSeVoc/gt/200_126784_000070_000000.wav
```