import requests
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

from config import utils_config as cfg


def download_file_from_google_drive(id, destination):
    print("Downloading ", destination.rpartition('/')[-1])
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                progress += len(chunk)
                pbar.update(progress - pbar.n)
                f.write(chunk)
        pbar.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transforms():
    transforms = cfg['transforms']

    ts = []
    for t in transforms:
        trans = t['transform']
        params = None
        if 'params' in t:
            params = t['params']

        if hasattr(A, trans):
            if params is not None:
                ts.append(getattr(A, trans)(**params))
            else:
                ts.append(getattr(A, trans)())

    transform = A.Compose([*ts, ToTensor()])
    return transform


def time_func(func):
    import time
    t1 = time.time()
    func()
    t2 = time.time()
    print("Time taken: %.2f %s" % (t2-t1, 'seconds'))
    return t2-t1


def main():
    pass


if __name__ == '__main__':
    main()
