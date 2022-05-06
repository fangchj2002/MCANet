#  Deep Speaker: speaker recognition system

Data Set: [LibriSpeech](http://www.openslr.org/12/)  
Reference paper: [MTANet: lightweight multi-scale temporal attention network for speaker recognition

  
This code was trained on librispeech-train-clean dataset, tested on librispeech-test-clean dataset. In my code, librispeech dataset shows ~5% EER with CNN model.   
  
## About the Code
`train.py`    
This is the main file, contains training, evaluation and save-model function  
`models.py`    
The neural network used for the experiment. This file contains three models, CNN model (same with the paper’s CNN), GRU model (same with the paper's GRU), simple_cnn model. simple_cnn model has similar performance with the original CNN model, but the number of trained parameter dropped from 24M to 7M.   
`select_batch.py`    
Choose the optimal batch feed to the network. This is one of the cores  of this experiment.     
`triplet_loss.py`    
This is a code to calculate triplet-loss for network training. Implementation is the same as paper.     
`test_model.py`    
This is a code that evaluates (test) the model, in terms of EER...      
`eval_matrics.py`  
For calculating equal error rate, f-measure, accuracy, and other metrics    
`pretaining.py`    
This is for pre-training on softmax classification loss.     
`pre_process.py`    
Load the utterance, filter out the mute(过滤静音), extract the fbank feature and save the module in .npy format.(提取fbank特征，以.npy格式保存模块) 
  
## Experimental Results  
This code was trained on librispeech-train-clean dataset, tested on librispeech-test-clean dataset. In my code, librispeech dataset shows ~5% EER with CNN model. 
  
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/loss.png"  width="400" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/EER.png" width="400" ></div>  


 ## Simple Use
1. Preprare data.   
I provide the sample data in `audio/LibriSpeechSamples/` or you can download full  [LibriSpeech](http://www.openslr.org/12/)  data or prepare your own data.   
2. Preprocessing.  
Extract feature and preprocessing: `python preprocess.py`.    
3. Training.   
If you want to train your model with Triplet Loss: `python train.py`.    
If you want to pretrain with softmax loss first: `python pretraining.py` then `python train.py`.    
Note: If you want to pretrain or not, you need to set `PRE_TRAIN`(in `constants.py`) flag with `True` or `False`.   

4. Evaluation.  
Evaluate the model in terms of EER: `test_model.py`.    
Note: During training,  `train.py` also evaluates the model.     
5. Plot loss curve.    
Plot loss curve and EER curve with `utils.py`.  
```
import constants as c
from utils import plot_loss
loss_file=c.CHECKPOINT_FOLDER+'/losses.txt' # loss file path
plot_loss(loss_file)
```
