from copy import deepcopy
from typing import List, Optional
import torch.nn as nn

class TTAModelManager:
    def __init__(self, model: nn.Module, norm_module: Optional[nn.Module], cali: Optional[nn.Module]):
        self.models = [m for m in [model, norm_module, cali] if m is not None]
        self.model = model
        self.norm_module = norm_module
        self.cali = cali
        
        self._initial_states = {}

    def configure_adaptation(self, strategy_str: str) -> List[nn.Parameter]:
        self._freeze_all()

        modules_to_adapt = self._find_modules_to_adapt(strategy_str)
        for name, module in modules_to_adapt:
            module.requires_grad_(True)
        trainable_params = []
        for model in self.models:
            for param in model.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        return trainable_params

    def snapshot(self):
        self._initial_states = {
            model: deepcopy(model.state_dict()) for model in self.models
        }

    def reset(self):
        for model, state in self._initial_states.items():
            model.load_state_dict(deepcopy(state), strict=True)

    def train(self):
        for m in self.models: m.train()

    def eval(self):
        for m in self.models: m.eval()

    def _freeze_all(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(False)

    def _find_modules_to_adapt(self, strategy_str: str) -> List[tuple]:
        all_named_modules = []
        for model in self.models:
            all_named_modules.extend(list(model.named_modules()))

        if strategy_str == 'all':
            return all_named_modules
        
        targets = []
        for rule in strategy_str.split(','):
            rule = rule.strip()
            is_exact = '(exact)' in rule
            name_key = rule.replace('(exact)', '')
            
            if is_exact:
                targets.extend([(n, m) for n, m in all_named_modules if n == name_key])
            else:
                targets.extend([(n, m) for n, m in all_named_modules if name_key in n])
        
        if not targets:
            raise ValueError(f"No modules found matching strategy: {strategy_str}")
        return targets