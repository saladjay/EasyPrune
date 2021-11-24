from demo.replaceapi import run, runback, log
from forwardlog import *

import torch

def record_model_layer_call_sequenece(model, input_tensor):

    log.add_input_blob_id(int(id(input_tensor)))
    run()
    with torch.no_grad():
        model(input_tensor)
    runback()

    log.remove_uesless_blob()
    ori_state_dict = model.state_dict()

