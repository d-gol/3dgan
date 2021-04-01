Kubeflow implementation of 3D gans.

Includes script and yaml file, to run it as a TFJob from Katib.

Steps to run:

- GPU 
    - Copy content of the file https://github.com/recardoso/ISC_HPC_3DGAN/blob/main/kubeflow/gan3d_preprocess.yaml to kubeflow/katib/#/katib/hp
- TPU
    - Copy content of the file https://github.com/recardoso/ISC_HPC_3DGAN/blob/main/kubeflow/gan3d_TPU.yaml to kubeflow/katib/#/katib/hp
- Select hyperparameters. Since we deploy with Katib, set both min and max values of the parameters to be the same, so we run just one job.
    - --nb_epochs - number of epochs
    - --is_full_training - do we run full dataset or just a 512 examples. Set it to 0 when want to just test things, otherwise 1
    - --use_eos - read data from eos, use on local Kubeflow instance. On GCP, use 1
    - --batch_size - batch size
    - --use_autotune - always leave to 1
    - --do_profling - do tensorflow profiling or not
- At kubeflow/katib/#/katib/hp, click Deploy
- After each epoch, results are stored in the bucket s3://dejan 
    - Sending you s3cmd config file on Mattermoost
    - Track progress of the job by occasionally running s3cmd ls s3://dejan
    - Example of the metrics result file on the bucket
        -  s3.cern.ch/dejan/tfjob-id-b50e4863-7fa5-4512-a3a4-df0d8787fa68-epoch-4-2020-12-08-17:20:10.txt
-  Process results
    -  To read one or more new jobs: https://github.com/recardoso/ISC_HPC_3DGAN/blob/main/kubeflow/process_results/results_processing_add_jobs.ipynb
    -  To plot from the csv file https://github.com/recardoso/ISC_HPC_3DGAN/blob/main/kubeflow/process_results/plotting.ipynb
    

^ kubeflow = ip address of a Kubeflow instance
