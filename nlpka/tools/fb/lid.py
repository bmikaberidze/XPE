import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import os
import fasttext
import requests
from tqdm import tqdm
import nlpka.tools.fb as tools_fb
tools_fb_path = common.get_module_location(tools_fb)

class FB_LID:

    fb_lid_ka_label = '__label__ka'

    fb_lid_name = 'lid.176.bin'
    fb_lid_url = f'https://dl.fbaipublicfiles.com/fasttext/supervised-models'

    def __init__(self):

        self.fb_lid_path = os.path.join(tools_fb_path, self.fb_lid_name)
        self.fb_lid_url = f'{self.fb_lid_url}/{self.fb_lid_name}'

        if os.path.exists(self.fb_lid_path):
            print("Facebooks Language identification model exists.")
        else:
            print(f"Facebooks Language identification model not found. Downloading from {self.fb_lid_url}")
            self._download()
        
        self.model = fasttext.load_model(self.fb_lid_path)

    def _download(self):
        response = requests.get(self.fb_lid_url, stream=True)
        with open(self.fb_lid_path, 'wb') as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    file.write(chunk)
    
    def predict_ka(self, texts, accuracy = 0.99, progress = False):
        # Remove newline characters from each text
        texts = [ text.replace('\n', ' ') for text in texts ]
        texts_ka = []
        lids = self.model.predict(texts)
        def one_loop_cycle(i):   
            # print(lids[0][i][0], lids[1][i][0], texts[i])
            lid_lang = lids[0][i][0]
            lid_acc = lids[1][i][0]
            if lid_lang == self.fb_lid_ka_label and lid_acc >= accuracy:
                texts_ka.append(texts[i])
        if progress:
            for i in tqdm(range(len(texts))):
                one_loop_cycle(i)
        else:
            for i in range(len(texts)):
                one_loop_cycle(i)
        return texts_ka

if (__name__ == '__main__'):
    pass