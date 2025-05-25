import logging
import os
from src.cleaning.clean_data import DataCleaner
from src.features.build_features import FeatureBuilder
from src.models.train_model import WebAttackModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebAttackPipeline:
    def __init__(self):
        self.data_dir = 'data'
        self.raw_data = os.path.join(self.data_dir, 'csic_database.csv')
        self.processed_data = os.path.join(self.data_dir, 'processed', 'cleaned_dataset.csv')
        self.feature_data = os.path.join(self.data_dir, 'features', 'engineered_features.csv')
        self.model_dir = os.path.join('src', 'models')

    def run_cleaning_pipeline(self):
        logger.info('Starting data cleaning pipeline...')
        try:
            cleaner = DataCleaner(self.raw_data, self.processed_data)
            cleaner.clean_data()
            logger.info('Data cleaning completed successfully')
        except Exception as e:
            logger.error(f'Error in cleaning pipeline: {e}')
            raise

    def run_feature_engineering(self):
        logger.info('Starting feature engineering pipeline...')
        try:
            feature_builder = FeatureBuilder(self.processed_data, os.path.join(self.data_dir, 'features'))
            feature_builder.build_features()
            logger.info('Feature engineering completed successfully')
        except Exception as e:
            logger.error(f'Error in feature engineering pipeline: {e}')
            raise

    def run_model_training(self):
        logger.info('Starting model training pipeline...')
        try:
            model = WebAttackModel(self.feature_data, self.model_dir)
            model.load_data()
            model.prepare_data()
            model.build_model()
            model.train_model()
            model.evaluate_model()
            logger.info('Model training completed successfully')
        except Exception as e:
            logger.error(f'Error in model training pipeline: {e}')
            raise

    def run_full_pipeline(self):
        try:
            logger.info('Starting full web attack detection pipeline...')
            self.run_cleaning_pipeline()
            self.run_feature_engineering()
            self.run_model_training()
            logger.info('Full pipeline completed successfully')
        except Exception as e:
            logger.error(f'Pipeline failed: {e}')
            raise

def main():
    pipeline = WebAttackPipeline()
    try:
        pipeline.run_full_pipeline()
    except Exception as e:
        logger.error(f'Pipeline execution failed: {e}')

if __name__ == '__main__':
    main()