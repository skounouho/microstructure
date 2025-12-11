This repository organizes the code for training a CNN and SVC to classify microstructures produced by computational AM simulations.

# Data Preparation

From the ORNL dataset, the six VTI files must be placed in a `data` directory. The VTI files have the file name structure:

```
r{num_r}_P{num_p}_grainid_rgbz.vti
```

Once the files are ready, we can run the following command to process the data into a `data.npz` file.

```
python run.py data
```

The processed data will be downsampled to 64-by-64 from the original 401-by-401 resolution. Next, we can do the initial training for the CNN and SVC. To train the SVC, simply run,

```
python run.py train --type svc
```

and for the CNN,

```
python run.py train --type cnn
```

# Initial Results

Training the CNN on the dataset achieved a training accuracy of 99.6% and a validation accuracy of 99.1%. Using a SVC on the dataset achieved a training accuracy of 99.82% and a validation accuracy of 99.12%.

# Next Steps

I am interested in the impact of defects, in the form of circular occulsions, which may represent defects in the microstructure. It would be interesting to see, for a set range of defects, how increasing the fraction of defected images in the training data might lead to better/worse accuracy in validation.

To generate the defect datasets, I ran the following command 
```
python run.py data --defect {num_defects}
```
where 8, 16, 32, 64, 128 were chosen as the number of defects.