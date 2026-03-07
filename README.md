# visher
Voice Phishing detection using Rawnet

Im using python 3.10.11

## How to run:

pip install -r requirements.txt

Train_visher : python main.py --data_path ./LibriSeVoc --batch_size 4 --num_epochs 2 --model_save_path ./checkpoints
Train_voiceguard: python main.py --data_path ./LibriSeVoc --batch_size 64 --num_epochs 50 --lr 0.0001 --model_save_path ./checkpoints

// folder "gt" for real voice
// other 6 folder is ai generated voices
Evaluate Real voice: python eval.py --input_path ./LibriSeVoc/gt/250_142286_000031_000006.wav --model_path ./checkpoints/best_model.pth
Evaluate Fake voice: python eval.py --input_path ./LibriSeVoc/wavernn/696_92939_000008_000001_gen.wav --model_path ./checkpoints/best_model.pth

## After training, we can prune and quantize the model. 
Run in sequence:
1. python pruning.py
2. python quantization.py
3. python quantized_inference.py