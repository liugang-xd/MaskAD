import copy
import importlib
import torch.nn as nn



class ModelHelper(nn.Module):
    def __init__(self, cfg):
        super(ModelHelper, self).__init__()
        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def forward(self, input):
        input = copy.copy(input)

        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        return input

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

