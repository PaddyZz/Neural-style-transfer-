
from src.vgg19.config.configuration import EvaluateConfig
import tensorflow as tf 

def model_save(extractor, config:EvaluateConfig):
    if config.save_keras:
        extractor.save(config.keras_saved_model_dir)
    else: 
        tf.saved_model.save(extractor,config.tf_saved_model_dir)

def image_save(image, config:EvaluateConfig):
    image.save(config.image_saved_path)