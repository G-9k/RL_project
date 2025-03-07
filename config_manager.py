import yaml
import os
from pathlib import Path

class ConfigManager:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        config_path = Path(__file__).parent / 'config.yaml'
        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)
    
    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._load_config()
        return cls._config
    
    @classmethod
    def get_env_config(cls):
        return cls.get_config()['environment']
    
    @classmethod
    def get_agent_config(cls):
        return cls.get_config()['agent']
    
    @classmethod
    def get_training_config(cls):
        return cls.get_config()['training']
    
    @classmethod
    def get_experiment_config(cls, experiment_name):
        return cls.get_config()['experiments'][experiment_name]
    
    @classmethod
    def get_wandb_config(cls):
        return cls.get_config()['wandb']