import os
from dotenv import load_dotenv
load_dotenv()
env = os.environ

if (__name__ == '__main__'):

    key = 'PACKAGE_NAME'
    print(
        env[key], 
        os.environ[key], 
        os.getenv(key)
    )

    env[key] = 'nlpka2'
    print(
        env[key], 
        os.environ[key], 
        os.getenv(key)
    )
    
    os.environ[key] = 'nlpka3'
    print(
        env[key], 
        os.environ[key], 
        os.getenv(key)
    )