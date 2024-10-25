from src.vgg19.components.pretrain_model import *
from src.vgg19.components.train_model import *
from src.vgg19.components.data_ingestion import *
from src.vgg19.components.model_eval import *
from src.vgg19.components.args import *
from src.vgg19 import logger
from src.vgg19.config.configuration import ConfigurationManager
import tensorflow as tf 

STAGE_NAME ="DATA_INGESTION"
STAGE_NAME_ONE = "PRETRAIN_MODEL"
STAGE_NAME_TWO = "TRAIN_MODEL"
STAGE_NAME_THREE = "MODEL_EVALUATE"


try:
        args_cope()
        data_ingestion_config = ConfigurationManager().get_data_ingestion_config()
        pretrain_model_config = ConfigurationManager().get_prepare_base_model_config()
        train_model_config  = ConfigurationManager().get_training_config()
        eval_model_config = ConfigurationManager().get_evaluation_config()
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        content_path = tf.keras.utils.get_file('content.jpg', data_ingestion_config.content_image_URL)
        style_path = tf.keras.utils.get_file('style.jpg',data_ingestion_config.style_image_URL)
        content_image = load_img(content_path)
        style_image = load_img(style_path)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        logger.info(f">>>>>> stage {STAGE_NAME_ONE} started <<<<<<")
        extractor = StyleContentModel(train_model_config.style_layers, train_model_config.content_layers,pretrain_model_config)
        image = tf.Variable(content_image)
        logger.info(f">>>>>> stage {STAGE_NAME_ONE} completed <<<<<<\n\nx==========x")
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} started <<<<<<")
        opt = tf.keras.optimizers.Adam(learning_rate=train_model_config.learning_rate, beta_1=train_model_config.beta_1, epsilon=float(train_model_config.epsilon))
        train_model(image,opt,content_image,style_image, extractor,config=train_model_config)
        logger.info(f">>>>>> stage {STAGE_NAME_TWO} completed <<<<<<\n\nx==========x")
        logger.info(f">>>>>> stage {STAGE_NAME_THREE} started <<<<<<")
        model_save(extractor=extractor, config=eval_model_config)
        image_save(tensor_to_image(image),eval_model_config)
        logger.info(f">>>>>> stage {STAGE_NAME_THREE} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e