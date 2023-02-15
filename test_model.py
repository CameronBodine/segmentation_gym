
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

from skimage.transform import resize
from skimage.filters import threshold_otsu

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


###########
# Functions
def rescale(dat,
    mn,
    mx):
    '''
    rescales an input dat between mn and mx
    '''
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn


# #-----------------------------------
def seg_file2tensor_ND(f, TARGET_SIZE):
    """
    "seg_file2tensor(f)"
    This function reads a NPZ image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of npz
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    with np.load(f) as data:
        bigimage = data["arr_0"].astype("uint8")

    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage


# #-----------------------------------
def seg_file2tensor_3band(f, TARGET_SIZE):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    bigimage = imread(f)
    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage


#=================================================
def Precision(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision

#=================================================
def Recall(confusionMatrix):
    epsilon = 1e-6
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis = 0) + epsilon)
    return recall

#=================================================
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score

#=================================================
def IntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU

#=================================================
def MeanIntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU

#=================================================
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

#=================================================
def ConfusionMatrix(numClass, imgPredict, Label):
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength = numClass**2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

#=================================================
def OverallAccuracy(confusionMatrix):
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

#=================================================
def MatthewsCorrelationCoefficient(confusionMatrix):

    t_sum = tf.reduce_sum(confusionMatrix, axis=1)
    p_sum = tf.reduce_sum(confusionMatrix, axis=0)

    n_correct = tf.linalg.trace(confusionMatrix)
    n_samples = tf.reduce_sum(p_sum)

    cov_ytyp = n_correct * n_samples - tf.tensordot(t_sum, p_sum, axes=1)
    cov_ypyp = n_samples ** 2 - tf.tensordot(p_sum, p_sum, axes=1)
    cov_ytyt = n_samples ** 2 - tf.tensordot(t_sum, t_sum, axes=1)

    cov_ytyp = tf.cast(cov_ytyp,'float')
    cov_ytyt = tf.cast(cov_ytyt,'float')
    cov_ypyp = tf.cast(cov_ypyp,'float')

    mcc = cov_ytyp / tf.math.sqrt(cov_ytyt * cov_ypyp)
    if tf.math.is_nan(mcc ) :
        mcc = tf.constant(0, dtype='float')
    return mcc.numpy()

#=================================================
def mean_iou_np(y_true, y_pred, nclasses):
    iousum = 0
    y_pred = tf.one_hot(tf.argmax(y_pred, -1), nclasses)
    for index in range(nclasses):
        iousum += basic_iou(y_true[:,:,:,index], y_pred[:,:,:,index])
    return (iousum/nclasses).numpy()

#=================================================
def basic_dice_coef(y_true, y_pred):
    """
    dice_coef(y_true, y_pred)
    This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions
    INPUTS:
        * y_true: true masks, one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
        * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
            * Inputs are B*W*H*N tensors, with
                B = batch size,
                W = width,
                H = height,
                N = number of classes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * Dice score [tensor]
    """
    smooth = 10e-6
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice

#=================================================
def mean_dice_np(y_true, y_pred, nclasses):
    dice = 0
    #can't have an argmax in a loss
    for index in range(nclasses):
        dice += basic_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return (dice/nclasses).numpy()

#=================================================
def AllMetrics(numClass, imgPredict, Label):

    confusionMatrix = ConfusionMatrix(numClass, imgPredict, Label)
    OA = OverallAccuracy(confusionMatrix)
    FWIoU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIoU = MeanIntersectionOverUnion(confusionMatrix)
    f1score = F1Score(confusionMatrix)
    recall = Recall(confusionMatrix)
    precision = Precision(confusionMatrix)
    mcc = MatthewsCorrelationCoefficient(confusionMatrix)

    return {"OverallAccuracy":OA,
            "Frequency_Weighted_Intersection_over_Union":FWIoU,
            "MeanIntersectionOverUnion":mIoU,
            "F1Score":f1score,
            "Recall":recall,
            "Precision":precision,
            "MatthewsCorrelationCoefficient": mcc}


def do_seg_csb(f, M, metadatadict, MODEL, outDir, NCLASSES, N_DATA_BANDS, TARGET_SIZE, TESTTIMEAUG, WRITE_MODELMETADATA, OTSU_THRESHOLD):
    '''
    Cam's version of do_seg for model testing and evaluation.

    Based off segmentation_gym by Daniel Buscombe.
    https://github.com/Doodleverse/segmentation_gym
    Workflow combines do_seg() in doodleverse_utils.prediction_imports.do_seg() &
    segmentation_gym.train_model.plotcomp_n_metrics.
    '''

    fname = os.path.basename(f)

    if f.endswith("jpg"):
        segfile = f.replace(".jpg", "_predseg.png")
    elif f.endswith("png"):
        segfile = f.replace(".png", "_predseg.png")
    elif f.endswith("npz"):  # in f:
        segfile = f.replace(".npz", "_predseg.png")

    if WRITE_MODELMETADATA:
        metadatadict["input_file"] = f

    # Set output directory
    out_dir_path = outDir
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    # Set output filepath
    segfile = os.path.basename(segfile)
    segfile = os.path.join(out_dir_path, segfile)

    # Store prediction metadata
    if WRITE_MODELMETADATA:
        metadatadict["nclasses"] = NCLASSES
        metadatadict["n_data_bands"] = N_DATA_BANDS

    # Store model metrics
    IOUc = []; Dc=[]; Kc = []
    OA = []; MIOU = []; FWIOU = []
    F1 = []; P =[]; R = []; MCC=[]

    if NCLASSES == 2:

        if N_DATA_BANDS <= 3:
            image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
        else:
            image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

        image = standardize(image.numpy()).squeeze()

        if MODEL=='segformer':
            if N_DATA_BANDS == 1:
                image = np.dstack((image, image, image))
            image = tf.transpose(image, (2, 0, 1))

        try:
            if MODEL=='segformer':
                est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).logits
            else:
                est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).squeeze()
        except:
            if MODEL=='segformer':
                est_label = model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1).logits
            else:
                est_label = model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1).squeeze()

        est_label = est_label.astype('float32')

        if MODEL=='segformer':
            est_label = resize(est_label, (1, NCLASSES, TARGET_SIZE[0],TARGET_SIZE[1]), preserve_range=True, clip=True).squeeze()
            est_label = np.transpose(est_label, (1,2,0))

        e0 = resize(est_label[:, :, 0], (w, h), preserve_range=True, clip=True)
        e1 = resize(est_label[:, :, 1], (w, h), preserve_range=True, clip=True)

        est_label = (e1 + (1 - e0)) / 2

        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = np.dstack((e0,e1))
        del e0, e1

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        if OTSU_THRESHOLD:
            thres = threshold_otsu(est_label)
            # print("Class threshold: %f" % (thres))
            est_label = (est_label > thres).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = thres

        else:
            est_label = (est_label > 0.5).astype("uint8")
            if WRITE_MODELMETADATA:
                metadatadict["otsu_threshold"] = 0.5

        ####
        # Not finished

    else:  ###NCLASSES>2

        if N_DATA_BANDS <= 3:
            image, w, h, bigimage = seg_file2tensor_3band(f, TARGET_SIZE)
        else:
            image, w, h, bigimage = seg_file2tensor_ND(f, TARGET_SIZE)

        image = standardize(image.numpy()).squeeze()

        if MODEL=='segformer':
            if N_DATA_BANDS == 1:
                image = np.dstack((image, image, image))
            image = tf.transpose(image, (2, 0, 1))

        try:
            if MODEL=='segformer':
                est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).logits
            else:
                est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).squeeze()
        except:
            if MODEL=='segformer':
                est_label = model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1).logits
            else:
                est_label = model.predict(tf.expand_dims(image[:,:,0], 0), batch_size=1).squeeze()

        # est_label cannot be float16 so convert to float32
        est_label = est_label.astype('float32')

        if MODEL=='segformer':
            est_label = resize(est_label, (1, NCLASSES, TARGET_SIZE[0],TARGET_SIZE[1]), preserve_range=True, clip=True).squeeze()
            est_label = np.transpose(est_label, (1,2,0))
        else:
            est_label = resize(est_label, (w, h))


        if WRITE_MODELMETADATA:
            metadatadict["av_prob_stack"] = est_label

        softmax_scores = est_label.copy() #np.dstack((e0,e1))

        if WRITE_MODELMETADATA:
            metadatadict["av_softmax_scores"] = softmax_scores

        imgPredict = np.argmax(softmax_scores, -1)

    #################################
    # Get prediction accuracy metrics
    # Load numpy file
    npz = np.load(f)

    # Get ground truth label
    gt_lbl = npz['arr_1']
    npz.close()

    # Turn one-hot encode into a label
    label = np.argmax(gt_lbl, -1)

    # Calculate accuracy metrics
    out = AllMetrics(NCLASSES, imgPredict, label)

    OA.append(out['OverallAccuracy'])
    FWIOU.append(out['Frequency_Weighted_Intersection_over_Union'])
    MIOU.append(out['MeanIntersectionOverUnion'])
    F1.append(out['F1Score'])
    R.append(out['Recall'])
    P.append(out['Precision'])
    MCC.append(out['MatthewsCorrelationCoefficient'])

    iouscore = mean_iou_np(np.expand_dims(np.squeeze(gt_lbl), 0), np.expand_dims(np.squeeze(est_label), 0), NCLASSES)

    dicescore = mean_dice_np(np.expand_dims(np.squeeze(gt_lbl), 0), np.expand_dims(np.squeeze(est_label), 0), NCLASSES)

    # #one-hot encode
    nx,ny = imgPredict.shape
    lstack = np.zeros((nx,ny,NCLASSES))
    lstack[:,:,:NCLASSES+1] = (np.arange(NCLASSES) == 1+imgPredict[...,None]-1).astype(int)

    kl = tf.keras.losses.KLDivergence()
    kld = kl(tf.squeeze(gt_lbl), lstack).numpy()


    IOUc.append(iouscore)
    Dc.append(dicescore)
    Kc.append(kld)

    metrics_table = {}
    metrics_table['Dataset'] = fname
    metrics_table['OverallAccuracy'] = np.array(OA)
    metrics_table['Frequency_Weighted_Intersection_over_Union'] = np.array(FWIOU)
    metrics_table['MeanIntersectionOverUnion'] = np.array(MIOU)
    metrics_table['MatthewsCorrelationCoefficient'] = np.array(MCC)

    df_out1 = pd.DataFrame.from_dict(metrics_table)

    metrics_per_class = {}
    metrics_per_class['Dataset'] = fname
    for k in range(NCLASSES):
        metrics_per_class['F1Score_class{}'.format(k)] = np.array(F1)[:,k]
        metrics_per_class['Recall_class{}'.format(k)] = np.array(R)[:,k]
        metrics_per_class['Precision_class{}'.format(k)] = np.array(P)[:,k]

    df_out2 = pd.DataFrame.from_dict(metrics_per_class)


    ##############################
    # Plotting and metadata export
    # Plot prediction
    class_label_colormap = [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
    ]

    class_label_colormap = class_label_colormap[:NCLASSES]

    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    try:
        color_label = label_to_colors(
            imgPredict,
            bigimage.numpy()[:, :, 0] == 0,
            alpha=128,
            colormap=class_label_colormap,
            color_class_offset=0,
            do_alpha=False,
        )
    except:
        try:
            color_label = label_to_colors(
                imgPredict,
                bigimage[:, :, 0] == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )
        except:
            color_label = label_to_colors(
                imgPredict,
                bigimage == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )

    # Color the ground truth label
    color_gt_lbl = label_to_colors(
        label,
        bigimage[:,:,0]==0,
        alpha=128,
        colormap=class_label_colormap,
        color_class_offset=0,
        do_alpha=0
    )

    # Save prediction to png
    imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)

    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    # Save metadata
    segfile = segfile.replace("_predseg.png", "_res.npz")

    if WRITE_MODELMETADATA:
        metadatadict["grey_label"] = imgPredict
        metadatadict["grey_gt_label"] = label

        np.savez_compressed(segfile, **metadatadict)

    # Convert ground truth to

    # Plotting
    segfile = segfile.replace("_res.npz", "_overlay.png")

    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])

    plt.imshow(color_label, alpha=0.5)
    plt.axis("off")
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches="tight")
    plt.close("all")

    #### image - overlay side by side
    segfile = segfile.replace("_res.npz", "_image_overlay.png")

    plt.subplot(131)
    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])
    plt.axis("off")
    plt.title('Image', fontsize=6)


    plt.subplot(132)
    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])
    plt.imshow(color_gt_lbl, alpha=0.5)
    plt.axis("off")
    plt.title('Ground Truth', fontsize=6)


    plt.subplot(133)
    if N_DATA_BANDS <= 3:
        plt.imshow(bigimage, cmap='gray')
    else:
        plt.imshow(bigimage[:, :, :3])
    # if NCLASSES>2:
    plt.imshow(color_label, alpha=0.5)
    # elif NCLASSES==2:
    #     cs = plt.contour(est_label, [-99,0,99], colors='r')
    plt.axis("off")
    plt.title('Prediction', fontsize=6)
    # plt.show()
    plt.savefig(segfile, dpi=200, bbox_inches="tight")
    plt.close("all")

    return df_out1, df_out2



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
# for f in tqdm(sample_filenames):
#     try:
#         do_seg(f, M, metadatadict, MODEL, outDir,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD)
#     except:
#         print("{} failed. Check config file, and check the path provided contains valid imagery".format(f))

for f in tqdm(sample_filenames):
    # try:
    #     do_seg_csb(f, M, metadatadict, MODEL, outDir,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD)
    # except:
    #     print("{} failed. Check config file, and check the path provided contains valid imagery".format(f))
    df1, df2 = do_seg_csb(f, M, metadatadict, MODEL, outDir,NCLASSES,N_DATA_BANDS,TARGET_SIZE,TESTTIMEAUG, WRITE_MODELMETADATA,OTSU_THRESHOLD)

    if 'dfSamp' not in locals():
        dfSamp = df1
    else:
        dfSamp = pd.concat([dfSamp, df1], ignore_index=True)

    if 'dfClass' not in locals():
        dfClass = df2
    else:
        dfClass = pd.concat([dfClass, df2], ignore_index=True)

fold = os.path.split(outDir)[-1]

dfSamp.to_csv(os.path.join(outDir, fold+'_model_metrics_per_sample_test.csv'))
dfClass.to_csv(os.path.join(outDir, fold+'_model_metrics_per_sample_per_class_test.csv'))
