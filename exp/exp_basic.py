import importlib
import os
from pathlib import Path
import torch



class Exp_Basic(object):
    def __init__(self, args):
        self.args = args

        # -------------------------------------------------------
        #  Automatically generate model map
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # Use smart dictionary
        self.model_dict = LazyModelDict(model_map)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        return

    def _scan_models_directory(self):
        """
        Automatically scan all .py files in the models folder
        """
        model_map = {}
        models_dir = Path(__file__).resolve().parents[1] / 'models'

        # Iterate through all files in 'models' directory
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # Ignore __init__.py and non-.py files
                if filename.endswith('.py') and filename != '__init__.py':
                    # Remove .py extension to get module name
                    module_name = filename[:-3]

                    # Build full import path
                    full_path = f"models.{module_name}"

                    # loading dict: {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path

        return model_map


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if not torch.cuda.is_available() or self.args.gpu >= torch.cuda.device_count():
                raise ValueError(f'CUDA device {self.args.gpu} is unavailable')
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def pretrain(self):
        pass

    def test(self):
        pass
    
    def _load_checkpoint(self, setting):
        ckpt = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=True)
        )

class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")

        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...")
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class
