import os
import pandas as pd
import glob
from auracog_lib.config.auconfig import AuConfig
from auracog_utils import DEFAULT_CONFIG_NAME
from auracog_utils.repo import AuraRepository

# El script que descarga los datasets de Artifactory necesita conocer mis credenciales. Dichas credenciales se encuentran (metidas por mi a mano) en:
# /opt/aura/app/etc/aura-utils.cfg

#CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
CURRENT_PATH = '/home/kike/Documentos/data'

def read_kpi_log():

    filename = 'ES-RECOGNIZER-NLP-concat.csv.bz2'
    filepath = os.path.join(CURRENT_PATH, 'logs_recognizer', filename)

    if not os.path.exists(filepath):
        cfg = AuConfig(DEFAULT_CONFIG_NAME)
        repo = AuraRepository(cfg)
        repo_items = repo.index(as_table=True)
        colnames = next(repo_items)
        datasets = pd.DataFrame.from_records(repo_items, columns=colnames)
        names = list(datasets['Name'])
        logs_recognizer = [n for n in names if n.startswith('KPI-ES')]
        print("{}".format(logs_recognizer))

        # Download datasets
        for keyname in logs_recognizer:
            print("Donwloaded {}".format(repo.get(keyname, os.path.join(CURRENT_PATH, 'logs_recognizer'))))

        # Concat datasets
        print("Generating concatenated recognizer csv")
        file_path = os.path.join(CURRENT_PATH, 'logs_recognizer')

        # TODO Print size of datasets by month

        concat_file = pd.concat([pd.read_csv(f, encoding='utf-8', sep=',') for f in glob.glob(os.path.join(file_path, 'ES-RECOGNIZER-NLP-*.csv.bz2'))], ignore_index = True)
        concat_file.to_csv(os.path.join(file_path, filename), sep=',', encoding='utf-8', compression='bz2', index=False)


read_kpi_log()
