from aicovidvn.datasets import AICovidVNDataset
from aicovidvn.models.cider_model import CIdeRModel
from aicovidvn.learners.cider_learner import CIdeRLeaner

def train():
    
    train_dataset = AICovidVNDataset(
        root='./data/aivncovid-19',
        mode='train',
        eval_type='maj_vote',
        transform=
    )