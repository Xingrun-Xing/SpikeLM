import torch
import logging
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss

import numpy as np
logging.basicConfig(level=logging.INFO)

class Learner(object):
    def __init__(self, student_model, teacher_model):

        self.student_model = student_model
        self.loss_mse = MSELoss()
        self.temperature = 1.

    def __call__(self, batch):
        student_outputs = self.student_model(**batch)
        real_loss = student_outputs.loss
        total_loss = 0 + 0 + real_loss

        return {'total_loss':total_loss, 'cls_dist_loss':0, 'mlm_dist_loss':0, 'rep_dist_loss':0, 'real_loss':real_loss}

