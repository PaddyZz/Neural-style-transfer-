from src.vgg19.constants import *
from src.vgg19.utils.common import read_yaml, create_directories
from src.vgg19.entity.config_entity import (DataIngestionConfig,
PretrainModelConfig,TrainModelConfig,EvaluateConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            content_image_URL = config.content_image_URL,
            style_image_URL = config.style_image_URL
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PretrainModelConfig:
        config = self.config.pretrain_model
        params = self.params
        create_directories([config.root_dir])

        pretrain_model_config = PretrainModelConfig(
            root_dir=Path(config.root_dir),
            weights_dir= Path(config.weights_dir)
        )

        return pretrain_model_config
    



    def get_training_config(self) -> TrainModelConfig:
        config = self.config.train_model
        params = self.params
        create_directories([
            Path(config.root_dir)
        ])

        train_model_config = TrainModelConfig(
            root_dir=Path(config.root_dir),
            learning_rate = params.LEARNING_RATE,
            beta_1 = params.BETA_1,
            epsilon = params.EPSILON,
            total_variation_weight = params.TOTAL_VARIATION_WEIGHT,
            epochs = params.EPOCHS,
            steps_per_epoch = params.STEPS_PER_EPOCH,
            style_weight = params.STYLE_WEIGHT,
            content_weight = params.CONTENT_WEIGHT,
            max_dim = params.MAX_DIM,
            content_layers= params.CONTENT_LAYERS,
            style_layers= params.STYLE_LAYERS
        )

        return train_model_config
    
    def get_evaluation_config(self) -> EvaluateConfig:
        config = self.config.evaluate_model
        params = self.params
        eval_config = EvaluateConfig(
            root_dir=Path(config.root_dir),
            saved_model_dir=Path(config.saved_model_dir),
            keras_saved_model_dir=Path(config.keras_saved_model_dir),
            tf_saved_model_dir = Path(config.tf_saved_model_dir),
            save_keras=params.SAVE_KERAS,
            image_saved_dir = Path(config.image_saved_dir)
        )
        return eval_config