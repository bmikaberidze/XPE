
# from google.cloud import translate
# from google.cloud import translate_v2
# https://stackoverflow.com/questions/72375391/httpcore-exceptions-readtimeout-the-read-operation-timed-out-python-googletr
# https://cloud.google.com/translate/docs/reference/libraries/v2/python

# from google_trans_new import google_translator 

import time
from datetime import datetime
import googletrans # pip3 install googletrans==3.1.0a0

class Translator():

    def __init__(self) -> None:
        # print(googletrans.LANGUAGES) # possible languages
        self.translator = googletrans.Translator()
        # self.translator2 = google_translator()

    def gettext(self, text, src='en', dest='ka', sleep = 10):
        translation = self.translate(text, src, dest, sleep)
        return translation.text

    def translate(self, text, src='en', dest='ka', sleep = 10):

        try:
            return self.translator.translate(text, src=src, dest=dest)
            # translation = self.translator2.translate(text, lang_tgt=dest, lang_src=src)
            # return (translation)

        # exception occurs due to many requests
        except Exception as e:
            
            print(str(datetime.now()) + " Translator.gettext error: " + str(e))

            if sleep == 120: 
                return None

            time.sleep(sleep)
            return self.translate(text, src, dest, sleep+5)

    # def translate2(self, text="YOUR_TEXT_TO_TRANSLATE", project_id="YOUR_PROJECT_ID"):
    #     """Translating Text."""

    #     location = "global"

    #     parent = f"projects/{project_id}/locations/{location}"

    #     # Translate text from English to French
    #     # Detail on supported types can be found here:
    #     # https://cloud.google.com/translate/docs/supported-formats
    #     response = self.client.translate_text(
    #         request={
    #             "parent": parent,
    #             "contents": [text],
    #             "mime_type": "text/plain",  # mime types: text/plain, text/html
    #             "source_language_code": "en-US",
    #             "target_language_code": "ka",
    #         }
    #     )

    #     # Display the translation for each input text provided
    #     for translation in response.translations:
    #         print("Translated text: {}".format(translation.translated_text))
