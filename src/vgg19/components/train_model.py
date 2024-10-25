import time
import tensorflow as tf
import matplotlib.pyplot as plt
from src.vgg19.components.data_ingestion import tensor_to_image
from src.vgg19.config.configuration import TrainModelConfig
from src.vgg19.components.pretrain_model import clip_0_1,style_content_loss,StyleContentModel


def train_step(image,opt,content_image, style_image, extractor:StyleContentModel, config:TrainModelConfig):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    loss = style_content_loss(outputs, style_targets,content_targets,extractor.num_content_layers,extractor.num_style_layers,config)
    loss += config.total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

def train_model(image,opt,content_image, style_image, extractor:StyleContentModel, config:TrainModelConfig):
    epochs = config.epochs
    steps_per_epoch = config.steps_per_epoch
    step = 0
    start = time.time()
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image,opt,content_image, style_image, extractor=extractor, config=config)
            print(".", end='', flush=True)
        print("Train step: {}".format(step))
        plt.imshow(tensor_to_image(image)) 
        end = time.time()
        print("Total time: {:.1f}".format(end-start))