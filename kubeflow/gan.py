#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
print(tf.__version__)
print(os.listdir('/tmp'))
print(os.listdir('/model_outputs'))
print(os.system('klist'))
print(os.listdir('/eos/user/d/dgolubov/tfrecords'))

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
import uuid
import json
print('TF_CONFIG', str(os.environ['TF_CONFIG']))


# In[2]:


GLOBAL_BATCH_SIZE = 64
nb_epochs = 3 #60 #Total Epochs
batch_size = 64 #batch size
latent_size = 256 #latent vector size
verbose = True
datapath = '/eos/user/d/dgolubov/tfrecords/public/*.tfrecords'# Data path
outpath = '/eos/user/d/dgolubov/tfresults/'# training output
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

tlab = False


# ## Models

# In[3]:


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
    add_loss = Lambda(count, arguments={'daxis':daxis2})(inv_image) # loss for bin counts
    Model(inputs=[image], outputs=[fake, aux, ang, ecal, add_loss]).summary() #removed add_loss
    return Model(inputs=[image], outputs=[fake, aux, ang, ecal, add_loss]) #removed add_loss


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


# ## Initialization 

# In[4]:


config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
main_session = tf.compat.v1.InteractiveSession(config=config)

WeightsDir = outpath + 'weights/3dgan_weights_' + name
pklfile = outpath + 'results/3dgan_history_dejan_' + name + '.pkl'# loss history
resultfile = outpath + 'results/3dgan_analysis' + name + '.pkl'# optimization metric history   
prev_gweights = ''#outpath + 'weights/' + params.prev_gweights
prev_dweights = ''#outpath + 'weights/' + params.prev_dweights

#loss_weights=[params.gen_weight, params.aux_weight, params.ang_weight, params.ecal_weight]
loss_weights=[3, 0.1, 25, 0.1, 0.1]
energies = [0, 110, 150, 190, 1]

#Define Strategy and models
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

print('batch_size', batch_size)

BATCH_SIZE_PER_REPLICA = batch_size

print('BATCH_SIZE_PER_REPLICA', str(BATCH_SIZE_PER_REPLICA))
print('strategy.num_replicas_in_sync', str(strategy.num_replicas_in_sync))

batch_size = batch_size * strategy.num_replicas_in_sync
print('batch_size', batch_size)

batch_size_per_replica=BATCH_SIZE_PER_REPLICA
print('batch_size_per_replica', batch_size_per_replica)


# In[5]:


#Gan util functions

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


# In[6]:


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


# In[7]:


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
    

Trainfiles, Testfiles = DivideFiles(datapath, f, datasetnames=["ECAL"], Particles =[particle])
#Trainfiles = ['/eos/user/d/dgolubov/tfrecords/public/Ele_VarAngleMeas_100_200_000.tfrecords']
#Testfiles = ['/eos/user/d/dgolubov/tfrecords/public/Ele_VarAngleMeas_100_200_001.tfrecords']
print("Trainfiles")
print(Trainfiles)

print("Testfiles")
print(Testfiles)

nb_Test = int(nEvents * f[1]) # The number of test events calculated from fraction of nEvents
nb_Train = int(nEvents * f[0]) # The number of train events calculated from fraction of nEvents

#create history and finish initiation
train_history = defaultdict(list)
test_history = defaultdict(list)
init_time = time.time()- start_init
analysis_history = defaultdict(list)
print('Initialization time is {} seconds'.format(init_time))


# In[8]:


#training functions

def Discriminator_Train_steps(discriminator, generator, dataset, batch_ind, nEvents, WeightsDir, pklfile, Trainfiles, daxis, daxis2, loss_ftn, combined, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
    global total_train_time
    
    # Get a single batch    
    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    # replica_context = tf.distribute.get_replica_context()
    # print(replica_context)
    # tf.print("Replica id: ", replica_context.replica_id_in_sync_group, " of ", replica_context.num_replicas_in_sync)
    # tf.print(energy_batch)


    #file_index +=1
    # Generate Fake events with same energy and angle as data batch
    noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
    generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
    #generated_images = generator.predict_on_batch(generator_ip, verbose=False)
    print('generator_ip.shape', str(generator_ip.shape))
    #generated_images = generator.predict_on_batch(generator_ip)
    generated_images = generator(generator_ip, training = False)

    #tf_config_str = str(os.getenv('TF_CONFIG'))
    #tfconfig_dict = dict(json.loads(tf_config_str))
    #with open('/eos/user/d/dgolubov/tfresults/debug-' + str(batch_ind) + '-worker-' + str(tfconfig_dict['task']['index']) + '.txt', 'w') as fd:
        #fd.write('Debugging\n')
        #fd.write('\ngenerated_images.shape' + str(generated_images.shape))
        #fd.write('\ngenerated_images\n\n' + str(generated_images.flatten()))

    #generated_images = tf.concat((generated_images, generated_images), axis=0)
    print('generated_images.shape', str(generated_images.shape))

    # Train discriminator first on real batch and then the fake batch
    print('image_batch.shape', str(image_batch.shape))
    print('energy_batch.shape', str(energy_batch.shape))
    print('ang_batch.shape', str(ang_batch.shape))
    print('ecal_batch.shape', str(ecal_batch.shape))
    print('add_loss_batch.shape', str(add_loss_batch.shape))

    print('discriminator.summary')
    discriminator.summary()

    start_train_time = time.time()
    real_batch_loss = discriminator.train_on_batch(image_batch, [BitFlip(np.ones(batch_size).astype(np.float32)), energy_batch, ang_batch, ecal_batch, add_loss_batch])
    total_train_time += time.time() - start_train_time

    start_train_time = time.time()
    fake_batch_loss = discriminator.train_on_batch(generated_images, [BitFlip(np.zeros(batch_size).astype(np.float32)), energy_batch, ang_batch, ecal_batch, add_loss_batch])
    total_train_time += time.time() - start_train_time

    #with open('/eos/user/d/dgolubov/tfresults/debug-' + str(batch_ind) + '-worker-' + str(tfconfig_dict['task']['index']) + '.txt', 'a') as fd:
        #fd.write('\nreal_batch_loss' + str(real_batch_loss))
        #fd.write('\nfake_batch_loss' + str(fake_batch_loss))

    return real_batch_loss, fake_batch_loss


def Generator_Train_steps(discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Trainfiles, daxis, daxis2, loss_ftn, combined, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
    # Get a single batch
    global total_train_time

    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    
    trick = np.ones(batch_size).astype(np.float32)
    gen_losses = []
    # Train generator twice using combined model
    for _ in range(2):
        noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
        generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1) # sampled angle same as g4 theta
        
        start_train_time = time.time()
        gen_losses.append(combined.train_on_batch(
            [generator_ip],
            [trick, tf.reshape(energy_batch, (-1,1)), ang_batch, ecal_batch, add_loss_batch]))
        total_train_time += time.time() - start_train_time

    generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]


    return generator_loss


# In[9]:


#testing functions

def Test_steps(discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Testfiles, daxis, daxis2, loss_ftn, combined, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):    
    # Get a single batch    
    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    # Generate Fake events with same energy and angle as data batch
    noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
    generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
    #generated_images = generator.predict_on_batch(generator_ip)
    generated_images = generator(generator_ip, training = False)
    #generated_images = tf.concat((generated_images, generated_images), axis=0)
    print('generated_images.shape', str(generated_images.shape))

    # concatenate to fake and real batches
    X = tf.concat((image_batch, generated_images), axis=0)
    y = np.array([1] * batch_size + [0] * batch_size).astype(np.float32)
    ang = tf.concat((ang_batch, ang_batch), axis=0)
    ecal = tf.concat((ecal_batch, ecal_batch), axis=0)
    aux_y = tf.concat((energy_batch, energy_batch), axis=0)
    add_loss= tf.concat((add_loss_batch, add_loss_batch), axis=0)

    disc_eval_loss = discriminator.evaluate( X, [y, aux_y, ang, ecal, add_loss], verbose=False, batch_size=batch_size)
    gen_eval_loss = combined.evaluate(generator_ip, [np.ones(batch_size), energy_batch, ang_batch, ecal_batch, add_loss_batch], verbose=False, batch_size=batch_size)

    return disc_eval_loss, gen_eval_loss


# In[ ]:


with strategy.scope():
    d=discriminator(xpower, dformat=dformat)
    g=generator(latent_size, dformat=dformat)
    
    # build the discriminator
    print('[INFO] Building discriminator')
    d.compile(
        optimizer=RMSprop(lr),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # build the generator
    print('[INFO] Building generator')
    g.compile(
        optimizer=RMSprop(lr),
        loss='binary_crossentropy'
    )

# build combined Model
latent = Input(shape=(latent_size, ), name='combined_z')   
fake_image = g( latent)
d.trainable = False
fake, aux, ang, ecal, add_loss = d(fake_image) #remove add_loss
with strategy.scope():
    combined = Model(
        inputs=[latent],
        outputs=[fake, aux, ang, ecal, add_loss], # remove add_loss
        name='combined_model'
    )
    combined.compile(
        optimizer=RMSprop(lr),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )
    
d.trainable = True # to allow updates to moving averages for BatchNormalization 

total_train_time = 0

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

    while nb_file < len(Trainfiles):
        #if index % 100 == 0:
        print('processed {} batches'.format(index + 1))
        print ('Loading Data from .....', Trainfiles[nb_file])

        # Get the dataset from the trainfile
        dataset = RetrieveTFRecord(Trainfiles[nb_file])
        #dataset = h5py.File(Trainfiles[0],'r') #to read h5py

        # Get the train values from the dataset
        dataset = GetDataAngleParallel(dataset, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
        nb_file+=1

        #create the dataset with tensors from the train values, and batch it using the global batch size
        dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)


        #Training
        #add Trainfiles, nb_train_batches, progress_bar, daxis, daxis2, loss_ftn, combined
        for batch_ind, batch in enumerate(dataset):

            #print(index)
            #gets the size of the batch as it will be diferent for the last batch
            this_batch_size = tf.shape(batch.get('Y')).numpy()[0]
            
            print('Training started...')
            #Discriminator Training
            real_batch_loss, fake_batch_loss = Discriminator_Train_steps(d, g, batch, batch_ind, nEvents, WeightsDir, pklfile,                 Trainfiles, daxis, daxis2, loss_ftn, combined,                 nb_epochs, this_batch_size, latent_size, loss_weights, lr, g_weights, d_weights, xscale, xpower,                 angscale, angtype, yscale, thresh, analyse, resultfile, energies, dformat, particle, verbose,                 warm, prev_gweights, prev_dweights)

            #if ecal sum has 100% loss(generating empty events) then end the training 
            if fake_batch_loss[3] == 100.0 and index >10:
                print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
                g.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                d.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                print ('real_batch_loss', real_batch_loss)
                print ('fake_batch_loss', fake_batch_loss)
                sys.exit()
            # append mean of discriminator loss for real and fake events 
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            generator_loss = Generator_Train_steps(d, g, batch, nEvents, WeightsDir, pklfile,                 Trainfiles, daxis, daxis2, loss_ftn, combined,                 nb_epochs, this_batch_size, latent_size, loss_weights, lr, g_weights, d_weights, xscale, xpower,                 angscale, angtype, yscale, thresh, analyse, resultfile, energies, dformat, particle, verbose,                 warm, prev_gweights, prev_dweights)

            epoch_gen_loss.append(generator_loss)
            index +=1


    #X_train, Y_train, ang_train, ecal_train = GetDataAngle(Trainfiles[0], xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
    epoch_train_time = time.time()-epoch_start
    print('Time taken by epoch{} was {} seconds.'.format(epoch, time.time()-epoch_start))

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


    # repeat till data is available
    while nb_file < len(Testfiles):

        print('processed {} batches'.format(index + 1))
        print ('Loading Data from .....', Testfiles[nb_file])

        # Get the dataset from the Testfile
        dataset = RetrieveTFRecord(Testfiles[nb_file])
        #dataset = h5py.File(Testfiles[0],'r') #to read h5py

        # Get the Test values from the dataset
        dataset = GetDataAngleParallel(dataset, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
        nb_file+=1

        #create the dataset with tensors from the Test values, and batch it using the global batch size
        dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)

        # Testing
        #add Testfiles, nb_test_batches, daxis, daxis2, X_train(??), loss_ftn, combined
        for batch in dataset:

            this_batch_size = tf.shape(batch.get('Y')).numpy()[0]

            disc_eval_loss, gen_eval_loss = Test_steps(d, g, batch, nEvents, WeightsDir, pklfile,                 Testfiles, daxis, daxis2, loss_ftn, combined,                 nb_epochs, this_batch_size, latent_size, loss_weights, lr, g_weights, d_weights, xscale, xpower,                 angscale, angtype, yscale, thresh, analyse, resultfile, energies, dformat, particle, verbose,                 warm, prev_gweights, prev_dweights)

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
    print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s} | {6:8s}'.format(
        'component', *d.metrics_names))
    print('-' * 65)
    ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}| {6:<10.2f}'
    print(ROW_FMT.format('generator (train)',
                            *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                            *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                            *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                            *test_history['discriminator'][-1]))

    # save weights every epoch                                                                                                                                                                                                                                                    
    g.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                            overwrite=True)
    d.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                overwrite=True)

    epoch_time = time.time()-test_start
    print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, epoch_time, WeightsDir))


    # save loss dict to pkl file
    pickle.dump({'train': train_history, 'test': test_history}, open(pklfile, 'wb'))

    print('train-loss:' + str(train_history['generator'][-1][0]))

    with open('/eos/user/d/dgolubov/tfresults/metrics_custom.txt', 'w') as metrics_wf:
        metrics_wf.write('only-train-epoch-time=' + str(total_train_time) + '\n')
        metrics_wf.write('train-epoch-time=' + str(epoch_train_time) + '\n')
        metrics_wf.write('test-epoch-time=' + str(epoch_time) + '\n')
        for i in range(len(d.metrics_names)):
            for model_kind in ['generator', 'discriminator']:
                metrics_wf.write(model_kind + '-train-' + d.metrics_names[i] + '=' + str(train_history[model_kind][-1][i]) + '\n')
                metrics_wf.write(model_kind + '-test-' + d.metrics_names[i] + '=' + str(test_history[model_kind][-1][i]) + '\n')

    with open('/model_outputs/metrics_custom.txt', 'w') as metrics_wf:
        metrics_wf.write('only-train-epoch-time=' + str(total_train_time) + '\n')
        metrics_wf.write('train-epoch-time=' + str(epoch_train_time) + '\n')
        metrics_wf.write('test-epoch-time=' + str(epoch_time) + '\n')
        for i in range(len(d.metrics_names)):
            for model_kind in ['generator', 'discriminator']:
                metrics_wf.write(model_kind + '-train-' + d.metrics_names[i] + '=' + str(train_history[model_kind][-1][i]) + '\n')
                metrics_wf.write(model_kind + '-test-' + d.metrics_names[i] + '=' + str(test_history[model_kind][-1][i]) + '\n')


    #--------------------------------------------------------------------------------------------
    #------------------------------ Analysis ----------------------------------------------------
    #--------------------------------------------------------------------------------------------


    # if a short analysis is to be performed for each epoch
#     if analyse:
#         print('analysing..........')
#         atime = time.time()
#         # load all test data
#         for index, dtest in enumerate(Testfiles):
#             if index == 0:
#                X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
#             else:
#                if X_test.shape[0] < nb_Test:
#                  X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
#                  X_test = np.concatenate((X_test, X_temp))
#                  Y_test = np.concatenate((Y_test, Y_temp))
#                  ang_test = np.concatenate((ang_test, ang_temp))
#                  ecal_test = np.concatenate((ecal_test, ecal_temp))
#         if X_test.shape[0] > nb_Test:
#            X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
#         else:
#            nb_Test = X_test.shape[0] # the nb_test maybe different if total events are less than nEvents      
#         var=gan.sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
#         result = gan.OptAnalysisAngle(var, g, energies, xpower = xpower, concat=2)
#         print('{} seconds taken by analysis'.format(time.time()-atime))
#         analysis_history['total'].append(result[0])
#         analysis_history['energy'].append(result[1])
#         analysis_history['moment'].append(result[2])
#         analysis_history['angle'].append(result[3])
#         print('Result = ', result)
#         # write analysis history to a pickel file
#         pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# In[ ]:


print(tf.__version__)


# In[ ]:





# # Notes
# 
# - Slow execution, run data loading as parallel pipeline steps
# - Training a bit slo

# In[ ]:




