# Naive CNN Model for Identifying AI Generated Faces

A very simplistic approach to a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) to verify if an image of a human face is AI generated or not. Special attention is given to training on high quality images and testing high quality images.

> [!WARNING]  
> This model is not working yet. It needs pre-processing for images or other contexts.

## About the Model

This model has 2 parameters: `real` or `synth`. It is trained on the following:

* FFHQ: real faces – https://github.com/NVlabs/ffhq-dataset
* SFHQ: synthetic faces – https://github.com/SelfishGene/SFHQ-dataset

This is a total of >160,000 images.

## Structure

> [!NOTE]  
> Model training is not synced.

* [saves](./saves/) – contains the saved trained `keras` model.
* [reports](./reports/) – results of previous runs and testing.
* [predict](./predict/) – the images used to test the model.
* [predict_modified](./predict_modified/) – modified versions of testing images (modifications made as a pre-process step to testing).
* [`get_predict.py`](./get_predict.py) – the runner for testing predefined images.

## To Use

1. Sync repository
2. Install Python [dependencies](./requirements.txt)
3. Use the [run notebook](./run.ipynb) to run tests.
4. Review [stats](./stats.ipynb) notebook

## Model Summary

```plain
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 1022, 1022, 64)    1792      
                                                                 
 max_pooling2d (MaxPooling2  (None, 511, 511, 64)      0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 509, 509, 128)     73856     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 254, 254, 128)     0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 252, 252, 256)     295168    
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 126, 126, 256)     0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 124, 124, 512)     1180160   
                                                                 
 global_average_pooling2d (  (None, 512)               0         
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 1)                 513       
...
Total params: 1551489 (5.92 MB)
Trainable params: 1551489 (5.92 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```