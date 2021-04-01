from __future__ import print_function
import os

import glob

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
import argparse
import sys
import h5py 
import numpy as np
import time
import math
import tensorflow as tf
import uuid

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import Progbar

from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Lambda,Dropout, Activation, Embedding)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import (UpSampling3D, Conv3D, ZeroPadding3D, AveragePooling3D)
from tensorflow.keras.models import Model, Sequential
import math

import json
import boto3
import datetime

#Configs
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
main_session = tf.compat.v1.InteractiveSession(config=config)

latent_size = 256 #latent vector size
verbose = True

outpath = '/eos/user/r/redacost/tfresults/'# training output
nEvents = 400000# maximum number of events used in training
ascale = 1 # angle scale
yscale = 100 # scaling energy«
xscale = 1
xpower = 0.85
angscale=1
analyse=False # if analysing
dformat='channels_first'
thresh = 0 # threshold for data
angtype = 'mtheta'
particle = 'Ele'
warm = False
lr = 0.001
events_per_file = 5000
name = 'gan_training'

g_weights='params_generator_epoch_'
d_weights='params_discriminator_epoch_'

job_id = uuid.uuid4()

tlab = False

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params')

    parser.add_argument('--nb_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--is_full_training', type=int, default=0, help='Load one file, or all files')
    parser.add_argument('--use_eos', type=int, default=0, help='Use EOS or s3 bucket to load files')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--use_autotune', type=int, default=0, help='Use autotune option for dataset processing')

    return parser

parser = get_parser()
args = parser.parse_args()

os.environ['S3_ENDPOINT'] = 'https://s3.cern.ch'
client = boto3.client('s3', endpoint_url='https://s3.cern.ch')

tf_config_str = os.environ.get('TF_CONFIG')
print(tf_config_str)
tf_config_dict  = json.loads(tf_config_str)
print(tf_config_dict)

nb_epochs = args.nb_epochs #60 #Total Epochs
is_full_training = args.is_full_training
use_eos = args.use_eos
batch_size = args.batch_size
use_autotune = args.use_autotune

outpath = './'# training output

# ## Models

# calculate sum of intensities
def ecal_sum(image, daxis):
    sum = K.sum(image, axis=daxis)
    return sum

# counts for various bin entries   
def count(image, daxis):
    limits=[0.05, 0.03, 0.02, 0.0125, 0.008, 0.003] # bin boundaries used
    bin1 = K.sum(tf.where(image > limits[0], K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin2 = K.sum(tf.where(tf.logical_and(image < limits[0], image > limits[1]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin3 = K.sum(tf.where(tf.logical_and(image < limits[1], image > limits[2]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin4 = K.sum(tf.where(tf.logical_and(image < limits[2], image > limits[3]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin5 = K.sum(tf.where(tf.logical_and(image < limits[3], image > limits[4]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin6 = K.sum(tf.where(tf.logical_and(image < limits[4], image > limits[5]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin7 = K.sum(tf.where(tf.logical_and(image < limits[5], image > 0.0), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin8 = K.sum(tf.where(tf.equal(image, 0.0), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bins = K.expand_dims(K.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1), axis=-1)
    return bins
                                        
# angle calculation 
def ecal_angle(image, daxis):
    image = K.squeeze(image, axis=daxis)# squeeze along channel axis
    
    # get shapes
    x_shape= K.int_shape(image)[1]
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1, 2, 3))# sum of events

    # get 1. where event sum is 0 and 0 elsewhere
    amask = tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
    masked_events = K.sum(amask) # counting zero sum events
    
    # ref denotes barycenter as that is our reference point
    x_ref = K.sum(K.sum(image, axis=(2, 3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5) , axis=1)# sum for x position * x index
    y_ref = K.sum(K.sum(image, axis=(1, 3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
    z_ref = K.sum(K.sum(image, axis=(1, 2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)
    x_ref = tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) , x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref) , y_ref/sumtot)
    z_ref = tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref), z_ref/sumtot)
    #reshape    
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)

    sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
        
    x = K.expand_dims(K.arange(x_shape), 0) # x indexes
    x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
    y = K.expand_dims(K.arange(y_shape), 0)# y indexes
    y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5
  
    #barycenter for each z position
    x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
    y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)
    x_mid = tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    #Angle Calculations
    z = (K.cast(K.arange(z_shape), dtype='float32') + 0.5)  * K.ones_like(z_ref) # Make an array of z indexes for all events
    zproj = K.sqrt(K.maximum((x_mid-x_ref)**2.0 + (z - z_ref)**2.0, K.epsilon()))# projection from z axis with stability check
    m = tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = tf.where(tf.less(z, z_ref),  -1 * m, m)# sign inversion
    ang = (math.pi/2.0) - tf.atan(m)# angle correction
    zmask = tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj) , zmask)
    ang = ang * zmask # place zero where zsum is zero
    
    ang = ang * z  # weighted by position
    sumz_tot = z * zmask # removing indexes with 0 energies or angles

    #zunmasked = K.sum(zmask, axis=1) # used for simple mean 
    #ang = K.sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

    ang = K.sum(ang, axis=1)/K.sum(sumz_tot, axis=1) # sum ( measured * weights)/sum(weights)
    ang = tf.where(K.equal(amask, 0.), ang, 100. * K.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events
    
    ang = K.expand_dims(ang, 1)
    return ang


def discriminator(power=1.0, dformat='channels_last'):
    K.set_image_data_format(dformat)
    if dformat =='channels_last':
        dshape=(51, 51, 25,1) # sample shape
        daxis=4 # channel axis 
        baxis=-1 # axis for BatchNormalization
        daxis2=(1, 2, 3) # axis for sum
    else:
        dshape=(1, 51, 51, 25) 
        daxis=1 
        baxis=1 
        daxis2=(2, 3, 4)
    image=Input(shape=dshape)

    x = Conv3D(16, (5, 6, 6), padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    dnn.summary()

    dnn_out = dnn(image)
    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    inv_image = Lambda(K.pow, arguments={'a':1./power})(image) #get back original image
    ang = Lambda(ecal_angle, arguments={'daxis':daxis})(inv_image) # angle calculation
    ecal = Lambda(ecal_sum, arguments={'daxis':daxis2})(inv_image) # sum of energies
    #add_loss = Lambda(count, arguments={'daxis':daxis2})(inv_image) # loss for bin counts
    Model(inputs=[image], outputs=[fake, aux, ang, ecal]).summary() #removed add_loss
    return Model(inputs=[image], outputs=[fake, aux, ang, ecal]) #removed add_loss


def generator(latent_size=256, return_intermediate=False, dformat='channels_last'):
    if dformat =='channels_last':
        dim = (9,9,8,8) # shape for dense layer
        baxis=-1 # axis for BatchNormalization
    else:
        dim = (8, 9, 9,8)
        baxis=1
    K.set_image_data_format(dformat)
    loc = Sequential([
        Dense(5184, input_shape=(latent_size,)),
        Reshape(dim),
        UpSampling3D(size=(6, 6, 6)),
        
        Conv3D(8, (6, 6, 8), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        ####################################### added layers 
        
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        ZeroPadding3D((1, 1, 0)),
        Conv3D(6, (3, 3, 5), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        #####################################  
        
        ZeroPadding3D((1, 1,0)),
        Conv3D(6, (3, 3, 3), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        
        Conv3D(1, (2, 2, 2),  padding='valid', kernel_initializer='glorot_normal'),
        Activation('relu')
    ])
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(inputs=[latent], outputs=[fake_image]).summary()
    return Model(inputs=[latent], outputs=[fake_image])

def compute_global_loss(labels, predictions, global_batch_size, loss_weights=[3, 0.1, 25, 0.1]):

    #can be initialized outside 
    binary_crossentropy_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    mean_absolute_percentage_error_object = tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)
    mae_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE) 

    binary_example_loss = binary_crossentropy_object(labels[0], predictions[0], sample_weight=loss_weights[0])
    mean_example_loss_1 = mean_absolute_percentage_error_object(labels[1], predictions[1], sample_weight=loss_weights[1])
    mae_example_loss = mae_object(labels[2], predictions[2], sample_weight=loss_weights[2])
    mean_example_loss_2 = mean_absolute_percentage_error_object(labels[3], predictions[3], sample_weight=loss_weights[3])
    
    binary_loss = tf.nn.compute_average_loss(binary_example_loss, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[0])
    mean_loss_1 = tf.nn.compute_average_loss(mean_example_loss_1, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[1])
    mae_loss = tf.nn.compute_average_loss(mae_example_loss, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[2])
    mean_loss_2 = tf.nn.compute_average_loss(mean_example_loss_2, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[3])
    
    return [binary_loss, mean_loss_1, mae_loss, mean_loss_2]

# ## Initialization

if not os.path.exists(outpath + 'weights/3dgan_weights_gan_training/'):
    os.makedirs(outpath + 'weights/3dgan_weights_gan_training/')

if not os.path.exists(outpath + 'results/3dgan_history_gan_training/'):
    os.makedirs(outpath + 'results/3dgan_history_gan_training/')

if not os.path.exists(outpath + 'results/3dgan_analysis_gan_training/'):
    os.makedirs(outpath + 'results/3dgan_analysis_gan_training/')

WeightsDir = outpath + 'weights/3dgan_weights_' + name
pklfile = outpath + 'results/3dgan_history_' + name + '.pkl'# loss history
resultfile = outpath + 'results/3dgan_analysis' + name + '.pkl'# optimization metric history   
prev_gweights = ''#outpath + 'weights/' + params.prev_gweights
prev_dweights = ''#outpath + 'weights/' + params.prev_dweights

#loss_weights=[params.gen_weight, params.aux_weight, params.ang_weight, params.ecal_weight]
loss_weights=[3, 0.1, 25, 0.1]
energies = [0, 110, 150, 190, 1]

#strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
tpu_address = os.environ["TPU_NAME"]
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)
print('TPU strategy created')

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BATCH_SIZE_PER_REPLICA = batch_size
batch_size = batch_size * strategy.num_replicas_in_sync
batch_size_per_replica=BATCH_SIZE_PER_REPLICA

#Divide files in train and test lists     
def DivideFiles(FileSearch="/data/LCD/*/*.h5", Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    print ("Found {} files. ".format(len(Files)))
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out

def BitFlip(x, prob=0.05):
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


#auxiliar functions

def hist_count(x, p=1.0, daxis=(1, 2, 3)):
    limits=np.array([0.05, 0.03, 0.02, 0.0125, 0.008, 0.003]) # bin boundaries used
    limits= np.power(limits, p)
    bin1 = np.sum(np.where(x>(limits[0]) , 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x<(limits[0])) & (x>(limits[1])), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x<(limits[1])) & (x>(limits[2])), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x<(limits[2])) & (x>(limits[3])), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x<(limits[3])) & (x>(limits[4])), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x<(limits[4])) & (x>(limits[5])), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x<(limits[5])) & (x>0.), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x==0, 1, 0), axis=daxis)
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    bins[np.where(bins==0)]=1 # so that an empty bin will be assigned a count of 1 to avoid unstability
    return bins

def GetDataAngleParallel(dataset, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1): 
    X=np.array(dataset.get('ECAL'))* xscale
    Y=np.array(dataset.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    if angtype in dataset:
        ang = np.array(dataset.get(angtype))[indexes]
    #else:  
      #ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=ecal[indexes]
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)

    Y = [[el] for el in Y]
    ang = [[el] for el in ang]
    ecal = [[el] for el in ecal]

    final_dataset = {'X': X,'Y': Y, 'ang': ang, 'ecal': ecal}

    return final_dataset

def RetrieveTFRecord(recorddatapaths):
    recorddata = tf.data.TFRecordDataset(recorddatapaths)

    #print(type(recorddata))

    
    retrieveddata = {
        'ECAL': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'ecalsize': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        #'bins': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'energy': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'eta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'mtheta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'sum': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'theta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, retrieveddata)

    parsed_dataset = recorddata.map(_parse_function)

    #return parsed_dataset

    #print(type(parsed_dataset))

    for parsed_record in parsed_dataset:
        dataset = parsed_record

    dataset['ECAL'] = tf.reshape(dataset['ECAL'], dataset['ecalsize'])

    dataset.pop('ecalsize')

    return dataset

def RetrieveTFRecordpreprocessing(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    #ds_size = sum(1 for _ in recorddata)

    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


    retrieveddata = {
        'X': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        #'ecalsize': tf.io.FixedLenSequencFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        'Y': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0), #float32
        'ang': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
        'ecal': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        data['X'] = tf.reshape(data['X'],[1,51,51,25])
        #print(tf.shape(data['Y']))
        data['Y'] = tf.reshape(data['Y'],[1])
        data['ang'] = tf.reshape(data['ang'],[1])
        data['ecal'] = tf.reshape(data['ecal'],[1])
        #print(tf.shape(data['Y']))
        return data

    print('Caching')
    if use_autotune:
        parsed_dataset = recorddata.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(batch_size, drop_remainder=True).repeat().with_options(options)
    else:
        parsed_dataset = recorddata.map(_parse_function).batch(batch_size, drop_remainder=True).cache().repeat().with_options(options)

    return parsed_dataset
    #return parsed_dataset, ds_size


#Compilation of models and definition of train/test files


start_init = time.time()
f = [0.9, 0.1] # train, test fractions 

loss_ftn = hist_count # function used for additional loss

# apply settings according to data format
if dformat=='channels_last':
   daxis=4 # channel axis
   daxis2=(1, 2, 3) # axis for sum
else:
   daxis=1 # channel axis
   daxis2=(2, 3, 4) # axis for sum
    

#Trainfiles, Testfiles = DivideFiles(datapath, f, datasetnames=["ECAL"], Particles =[particle])
if use_eos:
    if is_full_training:
        Trainfiles = [
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_000.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_001.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_002.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_003.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_004.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_005.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_006.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_007.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_008.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_009.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_010.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_011.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_012.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_013.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_014.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_015.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_016.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_017.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_018.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_019.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_020.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_021.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_022.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_023.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_024.tfrecords']
        Testfiles = [
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_025.tfrecords',\
                    '/eos/user/d/dgolubovtfrecordsprepro/Ele_VarAngleMeas_100_200_026.tfrecords',\
                    '/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_027.tfrecords']
    else:
        Trainfiles = ['/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_000.tfrecords']
        Testfiles = ['/eos/user/d/dgolubov/tfrecordsprepro/Ele_VarAngleMeas_100_200_001.tfrecords']
else:
    if is_full_training:
        Trainfiles = ['gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_000.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_001.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_002.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_003.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_004.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_005.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_006.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_007.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_008.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_009.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_010.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_011.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_012.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_013.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_014.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_015.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_016.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_017.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_018.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_019.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_020.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_021.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_022.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_023.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_024.tfrecords']
        Testfiles = ['gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_025.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_026.tfrecords',\
                    'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_027.tfrecords']
    else:
        Trainfiles = ['gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_000.tfrecords']
        Testfiles = ['gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_001.tfrecords']


print(Trainfiles)
print(Testfiles)

nb_Test = int(nEvents * f[1]) # The number of test events calculated from fraction of nEvents
nb_Train = int(nEvents * f[0]) # The number of train events calculated from fraction of nEvents

#create history and finish initiation
train_history = defaultdict(list)
test_history = defaultdict(list)
init_time = time.time()- start_init
analysis_history = defaultdict(list)
print('Initialization time is {} seconds'.format(init_time))


#Training

def Train_steps(dataset):
    # Get a single batch    
    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)
 
    # Generate Fake events with same energy and angle as data batch
    noise = tf.random.normal((batch_size_per_replica, latent_size-2), 0, 1)
    generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
    generated_images = generator(generator_ip, training=False)

    # Train discriminator first on real batch 
    fake_batch = BitFlip(np.ones(batch_size_per_replica).astype(np.float32))
    fake_batch = [[el] for el in fake_batch]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    with tf.GradientTape() as tape:
        predictions = discriminator(image_batch, training=True)
        real_batch_loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
    
    gradients = tape.gradient(real_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
    
    #------------Minimize------------
    #aggregate_grads_outside_optimizer = (optimizer._HAS_AGGREGATE_GRAD and not isinstance(strategy.extended, parameter_server_strategy.))
    gradients = optimizer_discriminator._clip_gradients(gradients)

    #--------------------------------
    
    optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

    #Train discriminato on the fake batch
    fake_batch = BitFlip(np.zeros(batch_size_per_replica).astype(np.float32))
    fake_batch = [[el] for el in fake_batch]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    with tf.GradientTape() as tape:
        predictions = discriminator(generated_images, training=True)
        fake_batch_loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
    gradients = tape.gradient(fake_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
    gradients = optimizer_discriminator._clip_gradients(gradients)
    optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights



    trick = np.ones(batch_size_per_replica).astype(np.float32)
    fake_batch = [[el] for el in trick]
    labels = [fake_batch, tf.reshape(energy_batch, (-1,1)), ang_batch, ecal_batch]

    gen_losses = []
    # Train generator twice using combined model
    for _ in range(2):
        noise = tf.random.normal((batch_size_per_replica, latent_size-2), 0, 1)
        generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1) # sampled angle same as g4 theta   

        with tf.GradientTape() as tape:
            generated_images = generator(generator_ip ,training= True)
            predictions = discriminator(generated_images , training=True)
            loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)

        gradients = tape.gradient(loss, generator.trainable_variables) # model.trainable_variables or  model.trainable_weights
        gradients = optimizer_generator._clip_gradients(gradients)
        optimizer_generator.apply_gradients(zip(gradients, generator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

        for el in loss:
            gen_losses.append(el)

    return real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3], fake_batch_loss[0], fake_batch_loss[1], fake_batch_loss[2], fake_batch_loss[3], \
            gen_losses[0], gen_losses[1], gen_losses[2], gen_losses[3], gen_losses[4], gen_losses[5], gen_losses[6], gen_losses[7]   

def Test_steps(dataset):    
    # Get a single batch    
    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    # Generate Fake events with same energy and angle as data batch
    noise = np.random.normal(0, 1, (batch_size_per_replica, latent_size-2)).astype(np.float32)
    generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
    generated_images = generator(generator_ip, training=False)

    # concatenate to fake and real batches
    X = tf.concat((image_batch, generated_images), axis=0)
    y = np.array([1] * batch_size_per_replica + [0] * batch_size_per_replica).astype(np.float32)
    ang = tf.concat((ang_batch, ang_batch), axis=0)
    ecal = tf.concat((ecal_batch, ecal_batch), axis=0)
    aux_y = tf.concat((energy_batch, energy_batch), axis=0)
    #add_loss= tf.concat((add_loss_batch, add_loss_batch), axis=0)

    y = [[el] for el in y]

    labels = [y, aux_y, ang, ecal]
    disc_eval = discriminator(X, training=False)
    disc_eval_loss = compute_global_loss(labels, disc_eval, batch_size, loss_weights=loss_weights)
    
    trick = np.ones(batch_size_per_replica).astype(np.float32) #original doest have astype
    fake_batch = [[el] for el in trick]
    labels = [fake_batch, energy_batch, ang_batch, ecal_batch]
    generated_images = generator(generator_ip ,training= False)
    gen_eval = discriminator(generated_images , training=False)#combined(generator_ip, training=False)
    
    gen_eval_loss = compute_global_loss(labels, gen_eval, batch_size, loss_weights=loss_weights)

    return disc_eval_loss[0], disc_eval_loss[1], disc_eval_loss[2], disc_eval_loss[3], gen_eval_loss[0], gen_eval_loss[1], gen_eval_loss[2], gen_eval_loss[3]



@tf.function
def distributed_train_step(dataset):

    gen_losses = []
    real_batch_loss_1, real_batch_loss_2, real_batch_loss_3, real_batch_loss_4, \
    fake_batch_loss_1, fake_batch_loss_2, fake_batch_loss_3, fake_batch_loss_4, \
    gen_batch_loss_1, gen_batch_loss_2, gen_batch_loss_3, gen_batch_loss_4, \
    gen_batch_loss_5, gen_batch_loss_6, gen_batch_loss_7, gen_batch_loss_8  = strategy.run(Train_steps, args=(next(dataset),))
    
    real_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_1, axis=None)
    real_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_2, axis=None)
    real_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_3, axis=None)
    real_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_4, axis=None)
    real_batch_loss = [real_batch_loss_1, real_batch_loss_2, real_batch_loss_3, real_batch_loss_4]

    fake_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_1, axis=None)
    fake_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_2, axis=None)
    fake_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_3, axis=None)
    fake_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_4, axis=None)
    fake_batch_loss = [fake_batch_loss_1, fake_batch_loss_2, fake_batch_loss_3, fake_batch_loss_4]


    gen_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_1, axis=None)
    gen_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_2, axis=None)
    gen_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_3, axis=None)
    gen_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_4, axis=None)
    gen_batch_loss = [gen_batch_loss_1, gen_batch_loss_2, gen_batch_loss_3, gen_batch_loss_4]

    gen_losses.append(gen_batch_loss)

    gen_batch_loss_5 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_5, axis=None)
    gen_batch_loss_6 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_6, axis=None)
    gen_batch_loss_7 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_7, axis=None)
    gen_batch_loss_8 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_8, axis=None)
    gen_batch_loss = [gen_batch_loss_5, gen_batch_loss_6, gen_batch_loss_7, gen_batch_loss_8]

    gen_losses.append(gen_batch_loss)

    return real_batch_loss, fake_batch_loss, gen_losses


@tf.function
def distributed_test_step(dataset):
    disc_test_loss_1, disc_test_loss_2, disc_test_loss_3, disc_test_loss_4, \
    gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4 = strategy.run(Test_steps, args=(next(dataset),))

    disc_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_1, axis=None)
    disc_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_2, axis=None)
    disc_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_3, axis=None)
    disc_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_4, axis=None)
    disc_test_loss = [disc_test_loss_1, disc_test_loss_2, disc_test_loss_3, disc_test_loss_4]

    gen_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_1, axis=None)
    gen_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_2, axis=None)
    gen_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_3, axis=None)
    gen_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_4, axis=None)
    gen_test_loss = [gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4]

    
    return disc_test_loss, gen_test_loss

with strategy.scope():
    discriminator=discriminator(xpower, dformat=dformat)
    generator=generator(latent_size, dformat=dformat)
    optimizer_discriminator = RMSprop(lr)
    optimizer_generator = RMSprop(lr)
  
    
print ('Loading Data')

dataset = RetrieveTFRecordpreprocessing(Trainfiles, batch_size)

dist_dataset = strategy.experimental_distribute_dataset(dataset)

dist_dataset_iter = iter(dist_dataset)

test_dataset = RetrieveTFRecordpreprocessing(Testfiles, batch_size)

test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

test_dist_dataset_iter = iter(test_dist_dataset)

#needs to change so it is not hard coded

if is_full_training:
    steps_per_epoch =int( 124987 // (batch_size))
    test_steps_per_epoch =int( 12340 // (batch_size))
    #steps_per_epoch =int( datasetsize // (batch_size))
    #test_steps_per_epoch =int( datasetsizetest // (batch_size))
else:
    steps_per_epoch =int( 512 // (batch_size))
    test_steps_per_epoch =int( 512 // (batch_size))

epoch_metrics = {}

# Start training
for epoch in range(nb_epochs):
    epoch_start = time.time()
    print('Epoch {} of {}'.format(epoch + 1, nb_epochs))


    #--------------------------------------------------------------------------------------------
    #------------------------------ Main Training Cycle -----------------------------------------
    #--------------------------------------------------------------------------------------------

    #Get the data for each training file

    nb_file=0
    epoch_gen_loss = []
    epoch_disc_loss = []
    index = 0
    file_index=0
    nbatch = 0

    print('Number of Batches: ', steps_per_epoch)
        
    for _ in range(steps_per_epoch):
        file_time = time.time()
        
        #Discriminator Training
        real_batch_loss, fake_batch_loss, gen_losses = distributed_train_step(dist_dataset_iter)

        #Configure the loss so it is equal to the original values
        real_batch_loss = [el.numpy() for el in real_batch_loss]
        real_batch_loss_total_loss = np.sum(real_batch_loss)
        new_real_batch_loss = [real_batch_loss_total_loss]
        for i_weights in range(len(real_batch_loss)):
            new_real_batch_loss.append(real_batch_loss[i_weights] / loss_weights[i_weights])
        real_batch_loss = new_real_batch_loss

        fake_batch_loss = [el.numpy() for el in fake_batch_loss]
        fake_batch_loss_total_loss = np.sum(fake_batch_loss)
        new_fake_batch_loss = [fake_batch_loss_total_loss]
        for i_weights in range(len(fake_batch_loss)):
            new_fake_batch_loss.append(fake_batch_loss[i_weights] / loss_weights[i_weights])
        fake_batch_loss = new_fake_batch_loss

        #if ecal sum has 100% loss(generating empty events) then end the training 
        if fake_batch_loss[3] == 100.0 and index >10:
            print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
            generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
            discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
            print ('real_batch_loss', real_batch_loss)
            print ('fake_batch_loss', fake_batch_loss)
            sys.exit()

        # append mean of discriminator loss for real and fake events 
        epoch_disc_loss.append([
            (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
        ])


        gen_losses[0] = [el.numpy() for el in gen_losses[0]]
        gen_losses_total_loss = np.sum(gen_losses[0])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(gen_losses[0])):
            new_gen_losses.append(gen_losses[0][i_weights] / loss_weights[i_weights])
        gen_losses[0] = new_gen_losses

        gen_losses[1] = [el.numpy() for el in gen_losses[1]]
        gen_losses_total_loss = np.sum(gen_losses[1])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(gen_losses[1])):
            new_gen_losses.append(gen_losses[1][i_weights] / loss_weights[i_weights])
        gen_losses[1] = new_gen_losses

        generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]

        epoch_gen_loss.append(generator_loss)
        batch_time = time.time()-file_time
        print('Time taken by batch', str(nbatch) ,' was', str(batch_time) , 'seconds.')
        epoch_metrics['time-batch-' + str(nbatch)] = batch_time
        nbatch += 1

    print('Time taken by epoch{} was {} seconds.'.format(epoch, time.time()-epoch_start))
    train_time = time.time() - epoch_start
    epoch_metrics['train-epoch-time'] = train_time

    discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

    #--------------------------------------------------------------------------------------------
    #------------------------------ Main Testing Cycle ------------------------------------------
    #--------------------------------------------------------------------------------------------

    #read first test file
    disc_test_loss=[]
    gen_test_loss =[]
    nb_file=0
    index=0
    file_index=0

    # Test process will also be accomplished in batches to reduce memory consumption
    print('\nTesting for epoch {}:'.format(epoch))
    test_start = time.time()



    # Testing
    #add Testfiles, nb_test_batches, daxis, daxis2, X_train(??), loss_ftn, combined
    for _ in range(test_steps_per_epoch):

        this_batch_size = 128 #can be removed (should)

        disc_eval_loss, gen_eval_loss = distributed_test_step(test_dist_dataset_iter)

        #Configure the loss so it is equal to the original values
        disc_eval_loss = [el.numpy() for el in disc_eval_loss]
        disc_eval_loss_total_loss = np.sum(disc_eval_loss)
        new_disc_eval_loss = [disc_eval_loss_total_loss]
        for i_weights in range(len(disc_eval_loss)):
            new_disc_eval_loss.append(disc_eval_loss[i_weights] / loss_weights[i_weights])
        disc_eval_loss = new_disc_eval_loss

        gen_eval_loss = [el.numpy() for el in gen_eval_loss]
        gen_eval_loss_total_loss = np.sum(gen_eval_loss)
        new_gen_eval_loss = [gen_eval_loss_total_loss]
        for i_weights in range(len(gen_eval_loss)):
            new_gen_eval_loss.append(gen_eval_loss[i_weights] / loss_weights[i_weights])
        gen_eval_loss = new_gen_eval_loss

        index +=1
        # evaluate discriminator loss           
        disc_test_loss.append(disc_eval_loss)
        # evaluate generator loss
        gen_test_loss.append(gen_eval_loss)


    #--------------------------------------------------------------------------------------------
    #------------------------------ Updates -----------------------------------------------------
    #--------------------------------------------------------------------------------------------


    # make loss dict 
    print('Total Test batches were {}'.format(index))
    discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
    discriminator_test_loss = np.mean(np.array(disc_test_loss), axis=0)
    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
    generator_test_loss = np.mean(np.array(gen_test_loss), axis=0)
    train_history['generator'].append(generator_train_loss)
    train_history['discriminator'].append(discriminator_train_loss)
    test_history['generator'].append(generator_test_loss)
    test_history['discriminator'].append(discriminator_test_loss)
    # print losses
    # print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s}'.format(
    #     'component', *discriminator.metrics_names))
    print(discriminator.metrics_names)
    print('-' * 65)
    ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}'
    print(ROW_FMT.format('generator (train)',
                            *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                            *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                            *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                            *test_history['discriminator'][-1]))

    # save weights every epoch                                                                                                                                                                                                                                                    
    generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                            overwrite=True)
    discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                overwrite=True)

    epoch_time = time.time()-test_start
    print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, epoch_time, WeightsDir))
    epoch_metrics['test-epoch-time'] = epoch_time
    
    # save loss dict to pkl file
    pickle.dump({'train': train_history, 'test': test_history}, open(pklfile, 'wb'))
    
    print('train-loss:' + str(train_history['generator'][-1][0]))

    metrics_names = ['loss', 'binary-loss', 'mean-loss-1', 'mae-loss', 'mean-loss-2']

    for i in range(len(metrics_names)):
        for model_kind in ['generator', 'discriminator']:
            epoch_metrics[model_kind + '-train-' + metrics_names[i]] = train_history[model_kind][-1][i]
            epoch_metrics[model_kind + '-test-' + metrics_names[i]] = test_history[model_kind][-1][i]

    epoch_metrics['batch-size'] = batch_size
    epoch_metrics['batch-size-per-replica'] = batch_size_per_replica
    epoch_metrics['num_replicas_in_sync'] = strategy.num_replicas_in_sync
    epoch_metrics['n_workers'] = len(tf_config_dict['cluster']['worker'])

    if tf_config_dict['task']['index'] == 0:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
        filename = 'tfjob-id-' + str(job_id) + '-epoch-' + str(epoch) + '-batchsize-' + str(batch_size) + '-' + str(timestamp) + '.txt'
        with open(filename, 'w') as f:
            for key, value in epoch_metrics.items():
                f.write(str(key) + '=' + str(value) + '\n')
        os.system('cp ' + filename + ' /model_outputs/metrics_custom.txt')
        client.upload_file(filename, 'dejan', filename)
                