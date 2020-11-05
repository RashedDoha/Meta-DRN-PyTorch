import requests
from torch.utils.model_zoo import tqdm

def download_file_from_google_drive(id, destination):
    print("Downloading ", destination.rpartition('/')[-1])
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

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
            if chunk: # filter out keep-alive new chunks
                progress += len(chunk)
                pbar.update(progress - pbar.n)
                f.write(chunk)
        pbar.close()

def main():
    pass

if __name__ == '__main__':
    main()
