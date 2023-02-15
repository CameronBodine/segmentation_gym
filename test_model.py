
# Written by Cameron S Bodine
# Based of segmentation gym workflows
# By Dan Buscombe

'''
Use previously trained model to evaluate accuracy on a test set.
'''

#########
# Imports
import sys,os, json
from tqdm import tqdm
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

############
# Parameters
npzCsv = sys.argv[1]
weights = sys.argv[2]
outDir = sys.argv[3]

df = pd.read_csv(npzCsv)
df['Files'] = df['Files'].apply(lambda r: os.path.normpath(r))
sample_filenames = list(df['Files'])
sample_filenames = sorted(sample_filenames)

# Do this because seg images in folder script allows loading multiple weight files
W = [weights]
M= []; C=[]; T = []
try:
    # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
    # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
    configfile = weights.replace('_fullmodel.h5','.json').replace('weights', 'config')
    with open(configfile) as f:
        config = json.load(f)
except:
    # Turn the .h5 file into a json so that the data can be loaded into dynamic variables
    configfile = weights.replace('.h5','.json').replace('weights', 'config')
    with open(configfile) as f:
        config = json.load(f)

for k in config.keys():
    exec(k+'=config["'+k+'"]')

# from doodleverse_utils.prediction_imports import *
# from tensorflow.python.client import device_lib
# physical_devices = tf.config.experimental.list_physical_devices('CPU')
# print(physical_devices)
#
# # for i in physical_devices:
# #     tf.config.experimental.set_memory_growth(i, True)
# # print(tf.config.get_visible_devices())

from doodleverse_utils.imports import *
from doodleverse_utils.model_imports import *
from doodleverse_utils.prediction_imports import *

##########################################
##### set up hardware
#######################################
if 'SET_PCI_BUS_ID' not in locals():
    SET_PCI_BUS_ID = False

SET_GPU = str(SET_GPU)
print(SET_GPU)

if SET_GPU != '-1':
    USE_GPU = True
    print('Using GPU')
else:
    USE_GPU = False
    print('Warning: using CPU - model training will be slow')

if len(SET_GPU.split(','))>1:
    USE_MULTI_GPU = True
    print('Using multiple GPUs')
else:
    USE_MULTI_GPU = False
    if USE_GPU:
        print('Using single GPU device')
    else:
        print('Using single CPU device')


if USE_GPU == True:

    ## this could be a bad idea - at least on windows, it reorders the gpus in a way you dont want
    if SET_PCI_BUS_ID:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ['CUDA_VISIBLE_DEVICES'] = SET_GPU

    from doodleverse_utils.imports import *
    from tensorflow.python.client import device_lib
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)

    if physical_devices:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from doodleverse_utils.imports import *
    from tensorflow.python.client import device_lib
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)


##########################
# Create and compile model

# Get the selected model based on the weights file's MODEL key provided
# create the model with the data loaded in from the weights file
print('.....................................')
print('Creating and compiling model...')

if MODEL =='resunet':
    model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    kernel_size=(KERNEL,KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    )
elif MODEL=='unet':
    model =  custom_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                    kernel_size=(KERNEL,KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                    dropout_type=DROPOUT_TYPE,
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                    )

elif MODEL =='simple_resunet':

    model = simple_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                dropout=DROPOUT,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                dropout_type=DROPOUT_TYPE,
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                filters=FILTERS,
                num_layers=4,
                strides=(1,1))

elif MODEL=='simple_unet':
    model = simple_unet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                dropout=DROPOUT,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                dropout_type=DROPOUT_TYPE,
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                filters=FILTERS,
                num_layers=4,
                strides=(1,1))

elif MODEL=='satunet':

    model = custom_satunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                kernel = (2, 2),
                num_classes=NCLASSES, #[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                activation="relu",
                use_batch_norm=True,
                dropout=DROPOUT,
                dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                dropout_type=DROPOUT_TYPE,
                use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                filters=FILTERS,
                num_layers=4,
                strides=(1,1))

elif MODEL=='segformer':
    id2label = {}
    for k in range(NCLASSES):
        id2label[k]=str(k)
    model = segformer(id2label,num_classes=NCLASSES)
    model.compile(optimizer='adam')

else:
    print("Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'")
    sys.exit(2)


if MODEL!='segformer':
    try:

        # Load in the model from the weights which is the location of the weights file
        model = tf.keras.models.load_model(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)

    except:
        # Load the metrics mean_iou, dice_coef from doodleverse_utils
        # Load in the custom loss function from doodleverse_utils
        model.compile(optimizer = 'adam', loss = dice_coef_loss(NCLASSES))#, metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])

        model.load_weights(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)

else:
        model.compile(optimizer = 'adam')

        model.load_weights(weights)

        M.append(model)
        C.append(configfile)
        T.append(MODEL)

# metadatadict contains the model name (T) the config file(C) and the model weights(W)
metadatadict = {}
metadatadict['model_weights'] = W
metadatadict['config_files'] = C
metadatadict['model_types'] = T


#####################################
#### read images
####################################

# # The following lines prepare the data to be predicted
# sample_filenames = sorted(glob(sample_direc+os.sep+'*.*'))
# if sample_filenames[0].split('.')[-1]=='npz':
#     sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.npz'))
# else:
#     sample_filenames = sorted(tf.io.gfile.glob(sample_direc+os.sep+'*.jpg'))
#     if len(sample_filenames)==0:
#         sample_filenames = sorted(glob(sample_direc+os.sep+'*.png'))

print('Number of samples: %i' % (len(sample_filenames)))

#####################################
#### run model on each image in a for loop
####################################
### predict
print('.....................................')
print('Using model for prediction on images ...')

#look for TTA config
if not 'TESTTIMEAUG' in locals():
    print("TESTTIMEAUG not found in config file(s). Setting to False")
    TESTTIMEAUG = False
if not 'WRITE_MODELMETADATA' in locals():
    print("WRITE_MODELMETADATA not found in config file(s). Setting to False")
    WRITE_MODELMETADATA = False
if not 'OTSU_THRESHOLD' in locals():
    print("OTSU_THRESHOLD not found in config file(s). Setting to False")
    OTSU_THRESHOLD = False

# Import do_seg() from doodleverse_utils to perform the segmentation on the images
for f in tqdm(sample_filenames):
    # try:
    #     do_seg(f, M, metadatadict, MODEL, outDir,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD)
    # except:
    #     print("{} failed. Check config file, and check the path provided contains valid imagery".format(f))

    do_seg(f, M, metadatadict, MODEL, outDir,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD)
