from aicovidvn.models.cider_model import CIdeRModel
from aicovidvn.learners.cider_learner import CIdeRLeaner

def train():
    depth_scale = 1
    nfft = 1024
    window_size = 8
    sample_rate = 24000

    model = CIdeRModel(
        dropout=False,
        depth_scale=depth_scale,
        input_shape=(int(1024 * nfft / 2048), int(94 * window_size * sample_rate / 48000))
    )

    learner = CIdeRLeaner(
        model=model
    )
    learner.train(
        root='./data/aivncovid-19',
        window_size=window_size, 
        n_nfft=nfft, 
        sample_rate=sample_rate,
        masking=True,
        pitch_shift=True,
        eval_type='maj_vote', 
        noise=True,
        batch_size=48,
        learning_rate=0.0001,
        n_epochs=100,
        shuffle=True, 
        save_dir='./models', 
        model_name='aicovidbn'
    )