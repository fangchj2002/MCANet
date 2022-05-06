DATASET_DIR = 'audio/LibriSpeechSamples/train-clean-100-train/'
TEST_DIR = 'audio/LibriSpeechSamples/test-clean/'
#IMAGE_DIR="audio/LibriSpeechSamples/train-clean-100-image/"
WAV_DIR = 'E:/test-data/'
KALDI_DIR = ''
BATCH_SIZE = 32   #must be even
TRIPLET_PER_BATCH =2

SAVE_PER_EPOCHS =1000
TEST_PER_EPOCHS = 200
CANDIDATES_PER_BATCH = 160# 18s per batch
TEST_NEGATIVE_No = 99


NUM_FRAMES = 160  # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
ALPHA = 0.1
HIST_TABLE_SIZE = 2
DATA_STACK_SIZE = 10

CHECKPOINT_FOLDER = 'checkpoints'
BEST_CHECKPOINT_FOLDER = 'checkpoints'
PRE_CHECKPOINT_FOLDER = 'pretraining_checkpoints'
GRU_CHECKPOINT_FOLDER = 'gru_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/train_losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/test_acc_eer.txt'
Test_LOSS_LOG = CHECKPOINT_FOLDER + '/test_losses.txt'

PRE_TRAIN = True
COMBINE_MODEL = False
