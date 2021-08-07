from aicovidvn.models.cider_model import CIdeRModel
from aicovidvn.learners.cider_learner import CIdeRLeaner

def infer():

    depth_scale = 1
    n_fft = 1024
    window_size = 8
    sample_rate = 24000

    model = CIdeRModel(
        dropout=False,
        depth_scale=depth_scale,
        input_shape=(int(1024 * n_fft / 2048), int(94 * window_size * sample_rate / 48000))
    )

    learner = CIdeRLeaner(model=model)
    learner.load_model(model_path='./models/aicovidvn.pt')

    out = learner.batch_inference(
        input='./data/aivncovid-19/test.csv',
        save_dir='./output',
        file_name='results.csv'
    )

infer()