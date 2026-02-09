# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4
lr = 10 ** log10_lr
epochs = 100
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 1
lamda_guide = 10
lamda_low_frequency = 0
device_ids = [1]

# Train:
batch_size = 4
cropsize = 224
betas = (0.5, 0.999)
weight_step = 100
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 16
shuffle_val = False
val_freq = 10


# Dataset

TRAIN_COVER_PATH = '/DIV2K/DIV2K_train_HR/'
TRAIN_SECRET_PATH = '/DIV2K/train-secret'
VAL_COVER_PATH = '/data/image'
VAL_SECRET_PATH = '/data/secret'
format_train = 'png'
format_val = 'png'
format_coco = 'jpg'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'g_loss', 'r_loss', 'f_loss', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = './model/'
checkpoint_on_error = True
SAVE_freq = 100

IMAGE_PATH = './images/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_cover_rev = IMAGE_PATH + 'cover-rev/'
IMAGE_PATH_resi_cover = IMAGE_PATH + 'resi-cover/'
IMAGE_PATH_resi_secret = IMAGE_PATH + 'resi-secret/'

# Load:
suffix = 'model.pt'
train_next = False
trained_epoch = 0
