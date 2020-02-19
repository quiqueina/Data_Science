"""# -*- coding: utf-8 -*-"""

# IMPORTS

import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
from auracog_lib.config.auconfig import AuConfig
from auracog_utils import DEFAULT_CONFIG_NAME
from auracog_utils.repo import AuraRepository

# CODE

# class _Preprocessing:
#
# 	def __init__(self):
# 	    return
# 	def __call__(self, *args, **kwargs):
#
# 	def main(self):
#
# 	pass
#
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



#######################################################################################################################
# DANIÉ
#######################################################################################################################
def parse_output(df):
	# Convert string to JSON dict
	output_df = df.OUTPUT.apply(lambda x: json.loads(x or '{}'))
	# JSON to DataFrame: Create a dataframe with each key as a new column
	output_df = pd.DataFrame(output_df.to_list())
	output_df = output_df[['score', 'intent', 'intents', 'entities']]  # re-order
	output_df.rename(columns={'score': 'OUTPUT_score_intent',
							  'intent': 'OUTPUT_intent',
							  'intents': 'OUTPUT_intents',
							  'entities': 'OUTPUT_entities'}, inplace=True)  # re-name 'score' columns

	return output_df


def parse_entity(output_df):
	# Unstack 'entities' list of entities: for each utterance, create a new entry per entity detected
	entities_df = output_df['OUTPUT_entities'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1,
																								 drop=True).to_frame(
		'ENTITIES_SINGLE_RAW')
	# JSON to DataFrame: Create a dataframe with each key as a new column
	entities_df = pd.DataFrame(entities_df['ENTITIES_SINGLE_RAW'].to_list(),
							   index=entities_df.index)  # explicitly pass the index because when doing .to_list we lose the indices, and we need to keep the reference of exploded/stacked entitites
	entities_df = entities_df[['type', 'entity', 'label', 'canon', 'start_index', 'end_index', 'score']]  # re-order
	entities_df.rename(columns={'score': 'ENTITY_score',
								'type': 'ENTITY_type',
								'entity': 'ENTITY_entity',
								'label': 'ENTITY_label',
								'canon': 'ENTITY_canon',
								'start_index': 'ENTITY_start_index',
								'end_index': 'ENTITY_end_index'}, inplace=True)  # re-name 'score' columns

	return entities_df


# Delete entries with repeated CORR_ID (due to bugs). In the future, this function won't be necessary
def delete_repeated_entries(df):
	# Fix (or bypass) for a bug that assigns the same CORR_ID to different samples (CORR_ID should be a unique identifier)
	corr_id_count = df.groupby('CORR_ID').CORR_ID.count()  # count the appearance of each CORR_ID
	CORR_ID_BUGs = corr_id_count[
		corr_id_count == 2].index.to_list()  # get a list with all CORR_IDs repeated exactly twice
	df = df.drop(df.loc[df.CORR_ID.isin(CORR_ID_BUGs)].index.to_list())  # delete all entries with these CORR_IDs

	# Fix (or bypass) for a bug that repeats n-times an entry with the same CORR_ID, but only the last is valid
	CORR_ID_REP = df.loc[df.duplicated('CORR_ID',
									   'last')].index.to_list()  # get a list with the indices of all repeated entries (but the 'last' of each repetition)
	df = df.drop(CORR_ID_REP)  # delete all entries with these CORR_IDs

	# Reset indices to avoid possible problems due to conversions from "list to dataframe", where indices are lost, and posterior concatenations of dataframes
	df = df.reset_index(drop=True)

	return df


# Standarizes columns for all dataframes according to latest version (v3 - May)
def standarize_columns(df, filename):
	df_date = filename.split("/")[-1].split("-")[3].split(".")[0]  # get dataframe's year-month

	df = df.rename(columns={'num_file': 'numFile'})  # rename num_file column to numFile
	df = df.reindex(columns=headers_canon)  # add missing columns to avoid columns mismatch when concatenating
	if df_date != '201905':  # for all log files but May (the most complete one according to header's canon)...
		df.userType = 'S'  # ...artificially fill all users as 'S' users (standard users)

	return df


# El script que descarga los datasets de Artifactory necesita conocer mis credenciales. Dichas credenciales se encuentran (metidas por mi a mano) en:
# /opt/aura/app/etc/aura-utils.cfg

# CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
CURRENT_PATH = '/home/daniel/Documentos/data'
FOLDER_NAME = 'logs_recognizer'
CONCAT_FOLDER = 'concat'

# Canonical column names to be shared by all dataframes
headers_canon = ['idx', 'RECOGNIZER_DT', 'RECOGNIZER_ID', 'AURA_ID', 'CHANNEL_CD', 'CORR_ID', 'STATUS_CD',
				 'REASON', 'VERSION_ID', 'DURATION_NU', 'SCORE_NU', 'INPUT', 'OUTPUT', 'INTENT',
				 'INTENT_RAW', 'ENTITIES', 'AURA_ID_GLOBAL', 'numFile', 'userType']


def read_kpi_log():
	# Path to save data files
	data_path = os.path.join(CURRENT_PATH, FOLDER_NAME)

	# Concatenated dataframe
	filename_concat = 'ES-RECOGNIZER-NLP-concat.csv.bz2'  # output name for concatenated dataframe
	filepath_concat = os.path.join(data_path, CONCAT_FOLDER, filename_concat)

	# Expanded-parsed-concatenated dataframe
	filename_concat_parsed = 'PARSED-ES-RECOGNIZER-NLP-concat.csv.bz2'  # output name for expanded-parsed-concatenated dataframe
	filepath_concat_parsed = os.path.join(data_path, CONCAT_FOLDER, filename_concat_parsed)

	# Download logs-datasets from Artifactory
	if not os.path.exists(filepath_concat):
		cfg = AuConfig(DEFAULT_CONFIG_NAME)
		repo = AuraRepository(cfg)
		repo_items = repo.index(as_table=True)
		colnames = next(repo_items)
		datasets = pd.DataFrame.from_records(repo_items, columns=colnames)
		names = list(datasets['Name'])
		logs_recognizer = [n for n in names if n.startswith('ES-NLP')]
		print("{}".format(logs_recognizer))

		# Download datasets
		print("\nDownloading datasets...")
		for keyname in tqdm(logs_recognizer):
			file = repo.get(keyname, data_path)
			print("Donwloaded {}".format(file))
		# TODO Print size of datasets by month

		# Concat datasets
		print("\nGenerating concatenated recognizer CSV (without duplicates)...")
		raw_files = sorted(glob(os.path.join(data_path, 'ES-RECOGNIZER-NLP-*.csv.bz2')))
		# concat_file = pd.concat([pd.read_csv(f, encoding='utf-8', dtype={'REASON': str, 'AURA_ID_GLOBAL': str}, sep=',') for f in raw_files], ignore_index=True)

		# INIT BLOCK: BLOCK TO BE removed when CORR_ID bugs are fixed (and UNCOMMENT "concat_file" line above)
		df2concat = []
		for f in tqdm(raw_files):
			df = pd.read_csv(f, encoding='utf-8', dtype={'REASON': str, 'AURA_ID_GLOBAL': str}, sep=',')
			df = delete_repeated_entries(df)  # LINE TO BE removed when CORR_ID bugs are fixed
			df = standarize_columns(df, f)  # LINE TO BE removed when all datasets have the same columns
			df2concat.append(df)
		concat_file = pd.concat(df2concat, ignore_index=True)
		# END BLOCK: BLOCK TO BE removed when CORR_ID bugs are fixed

		concat_file.to_csv(filepath_concat, sep=',', encoding='utf-8', compression='bz2', index=False)
	else:
		print("\nConcatenated recognizer CSV already exists!")

	# Process downloaded logs: parse OUTPUT column and generate a new column per key (and entity detected)
	if not os.path.exists(filepath_concat_parsed):
		print("\nParsing column 'OUTPUT'...")
		downloaded_files = sorted(glob(os.path.join(data_path, 'ES-RECOGNIZER-NLP-*.csv.bz2')))
		parsed_files = []  # list which will contain the names of the parsed files

		# Parse OUTPUT column from original monthly dataframes (necessary by month because the concatenated dataframe won't fit in memory in a short term)
		for fn in tqdm(downloaded_files):
			print("Processing {}...".format(fn.split("/")[-1]))
			df = pd.read_csv(fn, encoding='utf-8', dtype={'REASON': str, 'AURA_ID_GLOBAL': str}, keep_default_na=False,
							 sep=',')  # keep_default_na loads empty strings "as empty strings" instead of "as NaNs". This is necessary to then use json.loads() over 'OUTPUT' column
			df = delete_repeated_entries(df)  # LINE TO BE removed when CORR_ID bugs are fixed
			df = standarize_columns(df, fn)  # LINE TO BE removed when all datasets have the same columns

			# Create a new column per key of "OUTPUT-column JSON/dict"
			output_df = parse_output(df)

			# Create a new column per key of "OUTPUT_entities-column JSON/dict"
			entities_df = parse_entity(output_df)

			# Join both, "output_df" and "entities_df" in "df"
			df = df.join([output_df, entities_df], how='outer')

			# Save parsed DataFrame
			fp_out = os.path.join(data_path, 'PARSED-' + fn.split("/")[-1])  # output filepath
			parsed_files.append(fp_out)
			df.to_csv(fp_out, sep=',', encoding='utf-8', compression='bz2', index=False)

		# Save concatenated parsed logs
		print("\nGenerating concatenated recognizer CSV (without duplicates and expanding entities per utterance)...")
		concat_file = pd.concat(
			[pd.read_csv(fn, encoding='utf-8', dtype={'REASON': str, 'AURA_ID_GLOBAL': str}, sep=',') for fn in
			 tqdm(parsed_files)], ignore_index=True)
		concat_file.to_csv(filepath_concat_parsed, sep=',', encoding='utf-8', compression='bz2', index=False)
	else:
		print("\nExpanded-parsed-concatenated recognizer CSV already exists!")

	print("\nDONE! :)")


read_kpi_log()
#######################################################################################################################

from sklearn import preprocessing
import pandas as pd
import os


def format_logs(data_path):
    ## ENTITIES
    file_name_ent = 'PARSED-ES-RECOGNIZER-NLP-concat.csv.bz2'
    file_path_ent = os.path.join(data_path, file_name_ent)
    # DataFrame loading
    dfEnt = pd.read_csv(file_path_ent, encoding='utf-8', parse_dates=['RECOGNIZER_DT'], dtype={'REASON': str, 'AURA_ID_GLOBAL': str, 'INTENT_RAW': str}, sep=',') # necessary to specify dtype because, when parsing this columns, first values from which pandas infer types are empty, and so does not detect dtype properly
    dfEnt['DOMAIN'] = dfEnt.INTENT.str.split('.').str[0] # create a new column indicating the domain of the query/utterance
    dfEnt['RECOGNIZER_DT'] = dfEnt['RECOGNIZER_DT'].dt.tz_convert(tz='Europe/Madrid') # convert UTC time to local time

    ## INTENTS
    file_name_int = 'ES-RECOGNIZER-NLP-concat.csv.bz2'
    file_path_int = os.path.join(data_path, file_name_int)
    # DataFrame loading
    dfInt = pd.read_csv(file_path_int, encoding='utf-8', parse_dates=['RECOGNIZER_DT'], dtype={'REASON': str, 'AURA_ID_GLOBAL': str, 'INTENT_RAW': str}, sep=',')
    dfInt['DOMAIN'] = dfInt.INTENT.str.split('.').str[0] # create a new column indicating the domain of the query/utterance
    dfInt['RECOGNIZER_DT'] = dfInt['RECOGNIZER_DT'].dt.tz_convert(tz='Europe/Madrid') # convert UTC time to local time

    ########################################################################################################################

    ## ENTITIES
    dfEnt = dfEnt.loc[~(dfEnt.INTENT.isnull() & dfEnt.ENTITIES.isnull()) & ~dfEnt.AURA_ID_GLOBAL.isna()] # keep only those entries with intent, entity, or intent and entity (all except "empty intent AND empty entity"), and AURA_ID_GLOBAL not NaN
    dfEnt['ENTITY_type'] = dfEnt.ENTITY_type.fillna('intent_but_no_entity') # necessary for posteriory labelEncoder not tu crash (if it receives an empty string, throws an error) [empty entities happen in cases such as tv.on, common.greetings, etc...]. Also, in order to be able to cross fields with dfInt, fill ENTITY_type nulls instead of removing them.

    # Transform non-numerical labels to numerical labels
    le_ent = preprocessing.LabelEncoder() # create label encoder
    numeric_ent = le_ent.fit_transform(dfEnt.ENTITY_type) # fit label encoder with the categorical values/non-numerical labels, and transform all labels to numerical labels
    le_dom_ent = preprocessing.LabelEncoder() # create label encoder
    numeric_dom = le_dom_ent.fit_transform(dfEnt.DOMAIN) # fit label encoder with the categorical values/non-numerical labels, and transform all labels to numerical labels

    # Gather relevant timestamp info
    recognizer_dt = dfEnt.RECOGNIZER_DT # get timestamp for each entry
    time_info = [recognizer_dt.dt.year.rename('year'), recognizer_dt.dt.month.rename('month'), recognizer_dt.dt.weekday.rename('weekday'), recognizer_dt.dt.hour.rename('hour')] # extract year, month, weekday and hour of the day for each entry
    time_info = pd.concat(time_info, axis = 1) # create a dataframe with this information
    recognizer_dt = recognizer_dt.to_frame().join(time_info) # create a unique recognizer_dt dataframe with all relevant information relative to timestamp

    # Build entity-datetime dataframe
    ent_date_df = pd.DataFrame({'entity_num': numeric_ent, 'domain_num': numeric_dom}, index = dfEnt.index).\
        join([recognizer_dt, dfEnt.AURA_ID_GLOBAL, dfEnt.CORR_ID, dfEnt.ENTITY_type, dfEnt.DOMAIN, dfEnt.CHANNEL_CD, dfEnt.RECOGNIZER_ID, dfEnt.userType, dfEnt.RECOGNIZER_DT.dt.strftime('%Y-%m').rename('year_month'), dfEnt.RECOGNIZER_DT.dt.strftime('%Y-%W').rename('year_week')]).\
        rename(columns = {'RECOGNIZER_DT': 'datetime', 'AURA_ID_GLOBAL': 'aura_id_global', 'CORR_ID': 'corr_id', 'ENTITY_type': 'entity_label', 'DOMAIN': 'domain_label', 'CHANNEL_CD': 'channel', 'RECOGNIZER_ID': 'recognizer_id', 'userType': 'user_type'}) # create a dataframe with entities-datetime-aura ID global info // # strftime('%b %Y'): All days in a new year preceding the first Monday are considered to be in week 0.
    ent_date_df = ent_date_df[['recognizer_id', 'channel', 'corr_id', 'datetime', 'year_week', 'year_month', 'year', 'month', 'weekday', 'hour', 'aura_id_global', 'domain_label', 'domain_num', 'entity_label', 'entity_num', 'user_type']] # re-order


    ## INTENTS
    dfInt = dfInt.loc[~(dfInt.INTENT.isnull() & dfInt.ENTITIES.isnull()) & ~dfInt.AURA_ID_GLOBAL.isna()] # keep only those entries with intent, entity, or intent and entity (all except "empty intent AND empty entity"), and AURA_ID_GLOBAL not NaN

    # Transform non-numerical labels to numerical labels
    le_int = preprocessing.LabelEncoder() # create label encoder
    numeric_int = le_int.fit_transform(dfInt.INTENT) # fit label encoder with the categorical values/non-numerical labels, and transform all labels to numerical labels
    le_dom_int = preprocessing.LabelEncoder() # create label encoder
    numeric_dom = le_dom_int.fit_transform(dfInt.DOMAIN) # fit label encoder with the categorical values/non-numerical labels, and transform all labels to numerical labels

    # Gather relevant timestamp info
    recognizer_dt = dfInt.RECOGNIZER_DT # get timestamp for each entry
    time_info = [recognizer_dt.dt.year.rename('year'), recognizer_dt.dt.month.rename('month'), recognizer_dt.dt.weekday.rename('weekday'), recognizer_dt.dt.hour.rename('hour')] # extract year, month, weekday and hour of the day for each entry
    time_info = pd.concat(time_info, axis = 1) # create a dataframe with these information
    recognizer_dt = recognizer_dt.to_frame().join(time_info) # create a unique recognizer_dt dataframe with all relevant information relative to timestamp

    # Build intent-datetime dataframe
    int_date_df = pd.DataFrame({'intent_num': numeric_int, 'domain_num': numeric_dom}, index = dfInt.index).\
        join([recognizer_dt, dfInt.AURA_ID_GLOBAL, dfInt.CORR_ID, dfInt.INTENT, dfInt.DOMAIN, dfInt.CHANNEL_CD, dfInt.RECOGNIZER_ID, dfInt.userType, dfInt.RECOGNIZER_DT.dt.strftime('%Y-%m').rename('year_month'), dfInt.RECOGNIZER_DT.dt.strftime('%Y-%W').rename('year_week')]).\
        rename(columns = {'RECOGNIZER_DT': 'datetime', 'AURA_ID_GLOBAL': 'aura_id_global', 'CORR_ID': 'corr_id', 'INTENT': 'intent_label', 'DOMAIN': 'domain_label', 'CHANNEL_CD': 'channel', 'RECOGNIZER_ID': 'recognizer_id', 'userType': 'user_type'}) # create a dataframe with entities-datetime-aura ID global info // # strftime('%b %Y'): All days in a new year preceding the first Monday are considered to be in week 0.
    int_date_df = int_date_df[['recognizer_id', 'channel', 'corr_id', 'datetime', 'year_week', 'year_month', 'year', 'month', 'weekday', 'hour', 'aura_id_global', 'domain_label', 'domain_num', 'intent_label', 'intent_num', 'user_type']] # re-order

    return ent_date_df, int_date_df


# Path to logs' dataset
input_path = '/home/daniel/Documentos/data/logs_recognizer/concat'
ent_df, int_df = format_logs(input_path)

print("Done!")

#######################################################################################################################
# CARLOS & HUGO
#######################################################################################################################

import re
import unidecode
import num2words
import unicodedata
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from lemmatizer.token_lemmatizer import TokenLemmatizer
import json
import constants as co

def tokenize_text(text, word=True, tokenizer=None):
    """
    Tokenize a sentence using the nltk package
    :param text: input sentence
    :param word:
        if word, tokenize at word level (TreebankWordTokenizer by default)
        if not, tokenize at sentence level (PunktSentenceTokenizer by default)
    :param tokenizer: NLTK Tokenizer object, None by default
    :return: detected tokens
    Examples:
        >>> from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
        >>> text = 'This is an example sentence'
        >>> tokens = tokenize_text(text, word=True)
        >>> ['This', 'is', 'an', 'example', 'sentence']
        >>>
        >>> text = 'First example sentence. Second example sentence'
        >>> tokens = tokenize_text(text, word=False)
        >>> ['First example sentence.', 'Second example sentence']
    """
    if tokenizer is None and word:
        tokenizer = TreebankWordTokenizer()
    elif tokenizer is None and not word:
        tokenizer = PunktSentenceTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def remove_blanks(tokens):
    """
    Remove blanks in a list of tokens
    :param tokens: list of tokens
    :return: list of tokens without blanks
    Example:
        >>> tokens = ['This', ' is', '   an   ', 'example ', 'sentence']
        >>> remove_blanks(tokens)
        >>> ['This', 'is', 'an', 'an', 'example', 'sentence']
    """
    return [token.strip() for token in tokens]


def remove_accents(sentence):
    """
    Remove accents from all the characters (not single symbols) in a sentence
    :param sentence: sentence
    :return: sentence without accents
    Example:
        >>> import unidecode
        >>> sentence = 'Thís ïs àn éxample ¨¨¨*sentence'
        >>> remove_accents(sentence)
        >>> 'This is an example ```*sentence'
    """
    return unidecode.unidecode(sentence)


def remove_punctuation(sentence, regex=None):
    """
    Remove punctuation from an input sentence
    :param sentence: sentence
    :param regex: r'[^a-zA-Z0-9]' by default
    :return: sentence without punctuation
    Example
        >>> import re
        >>> sentence = 'This is an example ```*sentence'
        >>> remove_punctuation(sentence)
        >>> 'Th s  s  n  xample     sentence'
    """
    if regex is None:
        regex = r'[^a-zA-Z0-9]'
    else:
        regex = regex
    return re.sub(regex, r' ', sentence)


def normalize_unicode_data(sentence):
    sentence = unicodedata.normalize('NFKD', sentence).lower().encode('ascii', errors='ignore').decode('utf-8')
    return sentence


def remove_stopwords(tokens, stopwords_list=stopwords.words('spanish')):
    """
    Remove the stopwords in a list of tokens
    :param tokens: list of tokens
    :param stopwords_list: list of tokens. List of spanish stopwords from NLTK by default
    :return: list of tokens without stopwords
    Example:
        >>> from nltk.corpus import stopwords
        >>> tokens = ['Esta', 'es', 'una', 'frase', 'de', 'ejemplo']
        >>> remove_stopwords(tokens, stop)
        >>> ['Esta', 'frase', 'ejemplo']
    """
    return [token for token in tokens if token not in stopwords_list]


def num_to_words(tokens, lang='es', to='cardinal', with_decimals=False):
    """
    Converts numbers (like 42) to words (like forty-two) using the num2words package.
    It supports multiple languages and also different converters
    (see the documentation (https://pypi.org/project/num2words/)
    :param tokens: list of tokens
    :param lang: see the documentation to see the supported values
    :param to: Supported values: 'cardinal', 'ordinal', 'ordinal_num', 'year', 'currency'
    :param with_decimals: number with decimals
    :return: list of tokens with the numbers converted to words
    Example:
        >>> import num2words
        >>> tokens = ['1', '2', '100', '-1']
        >>> num_to_words(tokens)
        >>> ['uno', 'dos', 'cien', '-1']
        >>>
        >>> tokens = ['1992']
        >>> num_to_words(tokens, to='year')
        >>> ['mil novecientos noventa y dos']
    """
    if with_decimals:
        tokens = [num2words.num2words(token, lang=lang, to=to) if token.isdigit() else token for token in tokens]
    else:
        tokens = [num2words.num2words(int(token), lang=lang, to=to) if token.isdigit() else token for token in tokens]
    return tokens


def tokens_stemming(tokens, stemmer=PorterStemmer()):
    """
    Stem a list of tokens.
    :param tokens: list of tokens
    :param stemmer: stemmer to use. PorterStemmer by default
    :return: list of computed lexical roots, if not, the tokens
    Example:
        >>> from nltk.stem import PorterStemmer
        >>> tokens = ['pelicula', 'peliculas']
        >>> tokens_stemming(tokens)
        >>> ['pelicula', 'pelicula']
    """
    return [stemmer.stem(token) for token in tokens]


def normalize_censured_swearwords(sentence, alias='insult'):
    """
    Standardize all the censored swearwords with an alias
    :param sentence: input sentence
    :param alias: desired tag to replace the censored word
    :return: sentence with censored swearwords standardized with the alias
    """
    sentence = word_tokenize(sentence)
    for idx, word in enumerate(sentence):
        if '*' in word and word[0].isalpha():
            word[idx] = alias
    sentence = ' '.join(sentence)
    return sentence


def wordnet_token_lemmatization(tokens, lemmatizer=WordNetLemmatizer()):
    """
    Lemmatize a list of tokens. Does not work very well for spanish
    :param tokens: list of tokens
    :param lemmatizer:lemmatizer to use. WordNetLemmatizer to use
    :return: list of detected lexical roots, if not, the tokens
    Example:
        >>> from nltk.stem import WordNetLemmatizer
        >>> tokens = ['pelicula', 'peliculas']
        >>> tokens_lemmatization(tokens)
        >>> ['pelicula', 'peliculas']
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def token_lemmatization(sentence, language):
    """
    Lemmatize the words in a sentence using the lemmatizer module
    :param sentence: input sentence
    :param language: desired language. Only available language Spanish for the moment
    :return: sentence with the lemmas
    """
    lemmatizer = TokenLemmatizer(language)
    sentence = lemmatizer.lemmatize(sentence)
    return sentence


def extract_entities_from_recognizer_data(json_str):
    """
    >>> get_extraction_entities('{"score":0.4292148,"intent":"tef.int.es_ES.mp.tv.search","intents":[{"intent":"tef.int.es_ES.mp.tv.search","score":0.4292148}],"entities":[{"type":"tef.audiovisual_tvseries_title","entity":"el joven Sheldon","label":"","canon":"el joven sheldon","start_index":0,"end_index":16,"score":0.9999999999950252}]}')
    :param json_str:
    :return: (string_entity, entity_canon, entity_type)
    """
    values = []
    try:
        json_obj = json.loads(json_str)
        values = sorted([(' '.join([ut.convert_num_string(y) for y in x['entity'].split()]), '_'.join(x['canon'].split()), x['type'].split('.')[1]) for x in json_obj['entities']])
    except:
        pass

    return values


def extract_entities_from_ner(phrase, pkg, ner):
    """
    >>> from auracog_ner import Ner
    >>> from auracog_ner import VERSION
    >>> from pkg_resources import parse_version
    >>> from auracog_utils import DEFAULT_CONFIG_NAME
    >>> from auracog_utils import VERSION as UTILS_VERSION
    >>> from auracog_ner import Ner
    >>> from auracog_lib.config.auconfig import AuConfig
    >>> from auracog_utils.misc.notifier import Notifier
    >>> from auracog_utils.repo.model import AuraNlpModelRepository
    >>> cfg = AuConfig(DEFAULT_CONFIG_NAME)
    >>> ntf = Notifier(level='info')
    >>> nlp_repo = AuraNlpModelRepository(cfg, 'es-es', 'mh', notifier=ntf)
    >>> latest_version = nlp_repo.latest()
    >>> pkg = nlp_repo.open(latest_version)
    >>> metadata = pkg.release()
    >>> ner = Ner('crf', model_data=pkg)
    >>> #ner = Ner('gazetteer', model_data=pkg)
    >>> detect_entities_dictionary(sentence, pkg, ner)
    :param phrase:
    :param pkg:
    :param ner:
    :return:
    """
    # Open an standard (CRF) NER from the data in the NLP package
    dict_entities = None
    dict_entities = ner(phrase)['entities']
    return [(' '.join([num_to_words(y) for y in x['entity'].split()]), '_'.join(x['canon'].split()),
             x['type'].split('.')[1]) for x in dict_entities]


def transform_input_canon(phrase, list_entities):
    """
    >>> transform_input_canon('quiero ver el real madrid', [('real madrid', 'real_madrid', 'ent.audiovisual_sports_team')])
    :param phrase:
    :param list_entities:
    :return:
    """

    value = phrase.lower()
    for i in range(0, len(list_entities)):
        value = value.replace(list_entities[i][0].lower(), "_".join(list_entities[i][1].lower().split(" ")))
    return value


def transform_input_type(phrase, list_entities):
    """
    >>> transform_input_type('quiero ver el real madrid', [('real madrid', 'real_madrid', 'ent.audiovisual_sports_team')])
    :param phrase:
    :param list_entities:
    :return:
    """
    value=phrase
    for i in range(0, len(list_entities)):
        value = value.replace(ut.delete_punctuation(ut.convert_num_string(list_entities[i][1])).lower(), list_entities[i][2])
    return value


def set_schedule_time(hour):
    """
    >>> set_schedule_time(14)
    :param hour:
    :return:
    """
    if 6 < hour >= 23:
        return "early morning"
    if 6 >= hour < 12:
        return "morning"
    if 12 >= hour < 16:
        return "noon"
    if 16 >= hour < 20:
        return "afternoon"
    if 20 >= hour < 23:
        return "night"

def extract_topic_from_entities(obj, key):
    """
    >>> extract_topic_from_entities(co.dict_entities_topic, 'audiovisual_film_title')
    :param obj:
    :param key:
    :return:
    """
    arr = []
    results = []

    def extract(obj, arr, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (list)):
                    extract(v, arr, key)
                if (key in obj[k]):
                    arr.append(k)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    for entity in key:

        results += extract(obj, arr, entity)
    try:
        unique_result = list(set(results))[0]
    except Exception as e:
        unique_result='None'
    return unique_result
#######################################################################################################################


#######################################################################################################################
# JESÚS Y CARLOS
#######################################################################################################################
import os
import lemmatizer.constants as const


class TokenLemmatizer:
    def __init__(self, language):
        self.language = language
        self.lemmatizer = self.get_lemmas_dict()

    def get_lemmas_dict(self):
        lemmas_dict = {}
        with open(os.path.join(const.LEMMAS_PATH, const.LANGUAGE_FILES_MAPPING.get(self.language))) as f:
            for line in f:
                (key, val) = line.split()
                lemmas_dict[str(val)] = key
        return lemmas_dict

    def lemmatize(self, sentence):
        sentence = [self.lemmatizer.get(word, word) for word in sentence.split()]
        return sentence
#######################################################################################################################


#######################################################################################################################
# LISETTE
#######################################################################################################################

import pandas as pd
logs_path = '/home/lgm/Documents/Research/UXCI/Logs/ES-RECOGNIZER-NLP-201905-v3.csv.bz2'
week_log_output = '/home/lgm/Documents/Research/UXCI/Weeks/w8/logs/ES-RECOGNIZER-NLP-201905-v3-w8.csv.bz2'

df = pd.read_csv(logs_path)

# En el rango superior se pone un día más de la fecha deseada
mask = (df['RECOGNIZER_DT'] > '2019-05-20') & (df['RECOGNIZER_DT'] < '2019-06-01')
df = df[df['RECOGNIZER_ID']=='nlp-recognizer']
df = df[df['userType']=='S']


#######################################################################################################################
# GUISHE
#######################################################################################################################
import os
import sys
import logging
from datetime import datetime
from unidecode import unidecode
import emoji
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from ast import literal_eval as lit_eval

from config_utils import Config, Log
from phrase_modeling.phrase_modeling import PhraseModeling


class Normalizer(object):

    def __init__(self,
                 thr_entity=0.8,
                 thr_intent=0.8,
                 stopwords=stopwords.words('spanish'),
                 bg=False):
        """
        :param thr_entity: float, threshold value for filtering low entity classification scores
        :param thr_intent: float, threshold value for filtering low intent classification scores
        :param entity_filter: dict, Dictionary for mapping entity_type with a frequency of appearances value
        :param punctuation: List of strings, punctuation symbols
        :param stopwords: List of strings, stopwords provided by NLTK
        :param bg: boolean, whether or not to compute phrase modeling over the utterances
        """
        config_path = "/".join(os.path.abspath(__file__).split("/")[:-3]) + "/config/config.ini"
        paths, params, logs = self.get_params(config_path)
        self.thr_entity = thr_entity
        self.thr_intent = thr_intent
        self.stopwords = stopwords
        self.detokenizer = TreebankWordDetokenizer()
        self.punctuation = str.maketrans({key: " " for key in params["punctuation"]})
        self.entity_filter = params["entity_filter"]
        self.bg = bg
        self.not_matched_idx = []

        self.log_level = Normalizer.log_level(logs["log_level"])
        self.logger, self.log_normalizer, self.path_logs = self.__run_log(logs,
                                                                          paths,
                                                                          self.log_level)


    @staticmethod
    def get_params(config_path):
        """
        This function loads parameters from config.ini file
        """
        try:
            config = Config(config_path)

            paths = dict()
            params = dict()
            logs = dict()

            paths["general_path"] = "/".join(os.path.abspath(__file__).split("/")[:-1])
            paths["log_path"] = "/".join(os.path.abspath(__file__).split("/")[:-2]) +\
                                "/logs"

            params["punctuation"] = eval(config.get_config("FILTER_PARAMS", "PUNCTUATION"))
            params["entity_filter"] = eval(config.get_config("FILTER_PARAMS", "ENTITY_FILTER"))

            logs["log_level"] = config.get_config("LOGS", "LOG_LEVEL").replace('"', '')

            return paths, params, logs
        except Exception as e:
            print(e)


    def __run_log(self,
                  logs,
                  paths,
                  log_level):
        """
        This function loads log for the process.
        Creates a output file where is located the log file and others partial results are downloaded
        """
        try:

            output_file_name = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')).replace(" ", "_").replace(":", "_") + \
                               "_normalizer.log"
            paths["log_file_name"] = output_file_name
            logs["log_normalizer"] = paths["log_path"] + "/" + paths["log_file_name"]

            if not os.path.exists(paths["log_path"]):
                os.chmod(paths["general_path"], 0o777)
                os.makedirs(paths["log_path"])
                os.chmod(paths["log_path"], 0o777)

            log = Log("Normalizer_EmbeddingsEvaluation", log_level, logs["log_normalizer"])

            logger = log.set_log()
            return logger, logs["log_normalizer"], paths["log_path"]
        except Exception as e:
            print(e)


    @staticmethod
    def log_level(log_level):
        """
        Parses log level from config file.
        Now posible values are INFO and DEBUG.
        """
        try:
            if str(log_level).replace('"', '').lower() == "debug":
                return logging.DEBUG
            else:
                return logging.INFO
        except Exception as e:
            print(e)


    def normalize(self,
                  df,
                  tqdm_call=tqdm):
        '''
        :param df: pandas DataFrame
        :return df_proc: pandas DataFrame
        '''
        tqdm_call().pandas()

        # Filter rows with column RECOGNIZER_ID == 'aura-command-recognizer'
        self.logger.info("Filter rows with column RECOGNIZER_ID == 'aura-command-recognizer'")
        df.drop(df.loc[(df.RECOGNIZER_ID == 'aura-command-recognizer') | (df['OUTPUT'].astype(str) == 'nan')].index,
                axis=0,
                inplace=True)
        df.reset_index(inplace=True)

        # Filter rows by Intent classifier score
        # Create UNDER_THR column where 1 means that the thr has not been exceeded (candidates for being removed) and
        # 0 otherwise
        self.logger.info("Filter rows by Intent classifier score")
        df['UNDER_THR'] = np.where((df['SCORE_NU'] < self.thr_intent) | np.isnan(df['SCORE_NU']),
                                   1,
                                   0)
        df.drop(df.loc[(df['SCORE_NU'] < self.thr_intent) | (np.isnan(df['SCORE_NU']))].index,
                axis=0,
                inplace=True)
        df.reset_index(inplace=True)

        # Create entity map column
        self.logger.info("Creating Entity_map")
        df['ENTITY_MAP'] = df['OUTPUT'].progress_apply(lambda x: self.__get_extraction_entities(x))

        # Apply entity frequency filter
        self.logger.info("Entity frequency filter")
        df['ENTITY_MAP'] = self.__entity_freq_filt(df)

        # Lowercase and remove left and right spaces
        self.logger.info("Lowercasing and removing spaces")
        df['INPUT_PROC'] = df['INPUT'].str.strip().str.lower()

        # Remove diacritical marks, remove punctuation, replace emojis by special token
        self.logger.info("Remove diacritical marks, remove punctuation, replace emojis by special token")
        df['INPUT_PROC'] = \
            df['INPUT_PROC'].progress_apply(lambda x: re.sub(" +",
                                                             " ",
                                                             unidecode(str(x)).translate(self.punctuation))
            if emoji.emoji_count(str(x)) == 0 else unidecode(re.sub(r'\:(.+?)\:',
                                                                    '<emoji>',
                                                                    emoji.demojize(str(x).translate(self.punctuation)))
                                                             )
                                            )

        # Create INPUT_ENTITY column by replacing the entity_string by its entity_type
        # Create INPUT_PROC_BG_1 aggregating multiple-word entities in a single token (bigram, trigram, ...)
        # based on those recognized by NER model
        self.logger.info("Create INPUT_ENTITY ")
        df[['INPUT_ENTITY', 'INPUT_PROC_BG_1']] = df.progress_apply(lambda x: self.__replace_entities(x),
                                                                    axis=1)

        # If the entity was not matched in the input string because some exception occurred, update the UNDER_THR value
        # to 1
        df.loc[df['index'].isin(self.not_matched_idx), 'UNDER_THR'] = 1

        # Phrase modeling if requiered
        if self.bg:
            self.logger.info("Applying Phrase Modeling.")
            bgm = PhraseModeling()
            df = bgm.fit(df)

        return df


    def __replace_entities(self, df):
        """
        Match entity strings with those stored in the entity map for replacing the strings by its entity_type in the
         utterance.
        :param df: pandas DataFrame
        :return ut: string, utterance where the entities that appear in it have been replaced by its tag
        """
        # Processed utterance where we are going to search the entities stored as values in the ent_map
        input_proc = df['INPUT_PROC']
        # Dictionary of ent_type:[ent_str]
        ent_map = df['ENTITY_MAP']
        replace_dict = {}
        replace_dict_2 = {}
        try:
            # Append in a list every ent_str that match with the input_proc
            # TODO: re.sub("_", " ", v) --> v
            matches = [re.search(re.escape(re.sub("_", " ", v)), input_proc) for vlist in ent_map.values() for v in
                       vlist]
            # matches = [re.search(re.escape(v),input_proc) for vlist in ent_map.values() for v in vlist]
            if len(matches) > 0:
                # Construct the dictionary escaped(ent_str):ent_type
                for match in matches:
                    ent_str = input_proc[match.span()[0]:match.span()[1]]
                    for key in ent_map.keys():
                        # TODO: ent_str
                        if re.sub(" ", "_", ent_str) in ent_map[key]:
                            ent_tag = key
                        else:
                            continue
                    replace_dict[re.escape(ent_str)] = "[" + ent_tag + "]"
                    # TODO: replace_dict_2[re.escape(ent_str)] = ent_str
                    replace_dict_2[re.escape(ent_str)] = re.sub(" ", "_", ent_str)
                # Compile the patterns to find (ent_str)
                pattern = re.compile("|".join(replace_dict.keys()))
                pattern_2 = re.compile("|".join(replace_dict_2.keys()))
                # Replace in single step multiple matches
                input_entity = pattern.sub(lambda m: replace_dict[re.escape(m.group(0))], input_proc)
                input_proc_bg_1 = pattern_2.sub(lambda m: replace_dict_2[re.escape(m.group(0))], input_proc)
                return pd.Series([input_entity, input_proc_bg_1])
            else:
                return pd.Series([input_proc, input_proc])
        except Exception as e:
            self.logger.exception(e)
            # Store in a list the indexes which raised an exception for removing them
            self.not_matched_idx.append(df['index'])
            # Log the raised exceptions for future corrections
            self.logger.debug(str(df.idx) + '\t' + str(df.AURA_ID) + '\n' +
                              str(df.INPUT) + '\n' + df.OUTPUT + '\n' + str(df['ENTITY_MAP']) +
                              '\n' + df.INPUT_PROC + '\n\n')


    def __get_extraction_entities(self,
                                  json_str):
        """
        Generate dictionary with pairs: entity_type (key): list of entity_str (value)
        :param json_str: string, dictionary format as a string
        :param thr_entity: float, threshold for filtering low score entity values
        :return ent_map: dict, entity_tag(key) : list of entity strings (value)
        """
        ent_map = {}
        try:
            json_obj = json.loads(json_str)
            for d in json_obj['entities']:
                # Filter by entity score. If the thr is not exceeded, the token is not considered as a recognized entity
                if d['score'] >= self.thr_entity:
                    # Normalize the string of the entity
                    value = unidecode(
                        self.detokenizer.detokenize(d['entity'].split(" ")).strip().lower()).translate(
                        self.punctuation)
                    ent_type = d['type'].split(".")
                    if ent_type[0] == 'tef':
                        ent_type[0] = 'ent'
                    ent_type = ".".join(ent_type)
                    if ent_type in ent_map:
                        # TODO: Convert numeric strings to words avoiding the problem when it's treated as float
                        # TODO: ent_map[ent_type].append(value)
                        ent_map[ent_type].append(re.sub(" ", "_", value))
                    else:
                        # TODO: ent_map[ent_type] = [value]
                        ent_map[ent_type] = [re.sub(" ", "_", value)]
                else:
                    continue
        except Exception as e:
            self.logger.exception(e)

        return ent_map


    def build_entity_freq_dict(self,
                               df):
        """
        Construct entity frequency dictionary
        :param df: pandas DataFrame
        :return entities_dict: Dictionary
        """
        entities_dict = {}
        for d in df.ENTITY_MAP:
            if isinstance(d, str):
                d = lit_eval(d)
            for k, v in d.items():
                _dict = entities_dict.get(k, {})
                for val in v:
                    _num_occurrences = _dict.get(val, 0)
                    _dict[val] = _num_occurrences + 1
                entities_dict[k] = _dict
        return entities_dict

    def filter_entity_dict(self,
                           entity_dict):
        return {k: {k2: v2 for k2, v2 in v.items() if v2 > self.entity_filter[k]} for k, v in entity_dict.items()}


    def __entity_freq_filt(self,
                           df):
        """
        Construct frequency dictionary of entities and filter by a min_count parameter
        :param df: pandas DataFrame
        :return df: pandas DataFrame, modified input df removing from entity_map column entities which have not reached
         min_count threshold
        """
        # Build entity frequency dictionary
        entities_dict = self.build_entity_freq_dict(df)

        # Filter entity dict
        self.entities_dict_filt = self.filter_entity_dict(entities_dict)

        # Remove entites which have not reached min_count appearences
        new_entity_map = []
        for d in df['ENTITY_MAP']:
            new_d = {}
            for k, v in d.items():
                for e in v:  # for every value in the list of entities
                    if e in self.entities_dict_filt[k]:
                        if k in new_d:
                            new_d[k].append(e)
                        else:
                            new_d[k] = [e]
                    else:
                        continue
            new_entity_map.append(new_d)

        return new_entity_map
#######################################################################################################################
# VICTOR & JESÚS
#######################################################################################################################


import pandas as pd
import numpy as np
from typing import Text, Dict, List
import tqdm
import logging

from auracog_suggestions.utils import get_intent_code_mapping

logger = logging.getLogger()

_SESSION_LAPSES_MIN = [1, 2, 5, 10, 30, 60]
_NUM_MAX_INTENTS_PER_DOMAIN = 20

# Default intent codes, grouped by domain
# INTENT_CODES = {
#     None: 1,
#     'None': 1,
#     'tv.none': 2,
#
#     'common.greetings': 21,
#     'common.goodbyes': 22,
#     'common.thankyous': 23,
#     'common.help': 24,
#     'common.swearwords': 25,
#
#     'communications.add_contact': 41,
#     'communications.call': 42,
#     'communications.call_by_name': 43,
#     'communications.call_by_number': 44,
#     'communications.call_voicemail': 45,
#     'communications.check_calls': 46,
#     'communications.ignore_calls': 47,
#     'communications.phone_pick_up': 48,
#     'communications.redial': 49,
#
#     'communications.silent_mode': 51,
#     'communications.silent_mode_off': 52,
#     'communications.silent_mode_on': 53,
#
#     'domotics.light_off': 71,
#     'domotics.light_on': 72,
#
#     'tv.off': 91,
#     'tv.on': 92,
#
#     'ecommerce.add_to_wish_list': 111,
#
#     'wifi.get_access': 131,
#     'wifi.reset': 132,
#
#     'carousel.next': 151,
#     'carousel.previous': 152,
#
#     'tv.vod_epg_information': 171,
#     'carousel.info': 172,
#
#     'tv.search': 181,
#     'tv.content_get_info': 182,
#     'tv.question_time_loc': 183,
#     'tv.search_similar': 184,
#
#     'tv.channel_down': 191,
#     'tv.channel_up': 192,
#
#     'tv.display': 201,
#     'tv.launch': 202,
#     'tv.from_beginning': 203,
#     'tv.record': 204,
#     'tv.pause': 205,
#     'tv.resume': 206,
#
#     'tv.language_change': 211,
#     'tv.subtitles_remove': 212,
#     'tv.volume_down': 213,
#     'tv.volume_up': 214,
#     'tv.mute': 215,
#     'tv.unmute': 216
# }


INTENT_CODES = get_intent_code_mapping()


def _get_domain(intent_name: Text) -> Text:
    """
    Get the domain name from an intent name
    """
    if "." not in intent_name:
        return intent_name
    return intent_name[:intent_name.index(".")]


class RecognizerLogsFormatter(object):

    def __init__(self, session_lapses_min: List[int] = _SESSION_LAPSES_MIN):
        """
        :param session_lapses_min: List of lapses in minutes to group requests into the same session.
        """
        self.session_lapses_min = session_lapses_min

    def format_logs(self, file_path: Text, output_path: Text = None, output_format="csv", run_in_notebook=False,
                    translation_dict: Dict[Text, int] = INTENT_CODES) -> pd.DataFrame:
        """
        Format a Recognizer logs file into a pandas Dataframe.
        :param file_path:
        :param output_path: path to save the formatted data.
        :param output_format: Possible values: "csv"|"pick"
        :param run_in_notebook: True if this method is intended to be called within a jupyter notebook. False
        otherwise.
        :return Pandas Dataframe
        """
        df = pd.read_csv(file_path)

        # To numeric categorical values...
        df["n_AURA_ID"] = df["AURA_ID"].astype("category").cat.codes.apply(lambda x: float(x))
        df["n_AURA_ID_GLOBAL"] = df["AURA_ID_GLOBAL"].astype("category").cat.codes.apply(lambda x: float(x))
        # Convert dates
        df["RECOGNIZER_DT"] = pd.to_datetime(df["RECOGNIZER_DT"])
        # Change UTC time to Spain/Madrid UTC time
        df['RECOGNIZER_DT_CONVERT'] = df['RECOGNIZER_DT'].dt.tz_convert(tz='Europe/Madrid')
        # Build numerical timestamps
        df["TIMESTAMP"] = df['RECOGNIZER_DT_CONVERT'].values.astype(np.int64)
        # Add auxiliary ones column to make counts easier
        df["__ONES"] = np.ones((len(df)))
        # Add domain column
        df["DOMAIN"] = df["INTENT"].apply(lambda s: _get_domain(str(s)))
        # Convert dates
        # df["RECOGNIZER_DT"] = pd.to_datetime(df["RECOGNIZER_DT"])

        # Encode intents
        intent_encoder = IntentEncoder(df, translation_dict=translation_dict)
        df["INTENT_ENCODED"] = df.INTENT.apply(
            lambda s: intent_encoder.encode_intent(str(s)))

        # Order by date
        df = df.sort_values(by=["RECOGNIZER_DT_CONVERT"])

        # Add sessions data
        for lapse in self.session_lapses_min:
            df["SESSION_{}_MIN".format(lapse)] = np.ones(len(df)) * -1

        df_grouped_by_aura_id = df.groupby("AURA_ID_GLOBAL")
        if run_in_notebook:
            _tqdm = tqdm.tqdm_notebook
        else:
            _tqdm = tqdm.tqdm
        for g_name, _df in _tqdm(df_grouped_by_aura_id):
            for lapse in self.session_lapses_min:
                sessions = self._get_sessions_per_user(_df, lapse * 60)
                indices = _df.index
                for i, s in enumerate(sessions):
                    df.at[indices[i], "SESSION_{}_MIN".format(lapse)] = s

        # Save dataframe
        if output_path is not None:
            if output_format.lower() == "csv":
                df.to_csv(output_path)
            elif output_format.lower() in {"pick", "pickle"}:
                df.to_pickle(output_path)
            else:
                raise Exception("Wrong value for argument 'output_format': {}. Valid values are 'csv' | 'pick'".format(
                    output_format))
        return df

    def _get_sessions_per_user(self, df: pd.DataFrame, max_seconds: float = 1 * 60) -> List[int]:
        """
        Get an array of indexes to group related entries. Related entries are those whose time difference is
        less than or equal to max_seconds.
        :param df:
        :param max_seconds:
        :return
        """
        max_nano_seconds = max_seconds * 1e09
        _counter = 0
        res = []
        diff = df["TIMESTAMP"].diff()
        for v in diff.values:
            if v > max_nano_seconds:
                _counter += 1
            res.append(_counter)
        return res


class IntentEncoder(object):
    """
    This class encodes intent names into numerical values.
    The translation dictionary can be retrieved (if not specify in constructor) from translation_dict.
    """

    def __init__(self, df: pd.DataFrame, translation_dict: Dict[Text, int] = None,
                 num_max_intents_per_domain=_NUM_MAX_INTENTS_PER_DOMAIN):
        """
        :param df:
        :param translation_dict: Dictionary used for translation (intent_name -> numerical_value)
        :param num_max_intents_per_domain
        """
        self.num_max_intents_per_domain = num_max_intents_per_domain
        # Sorted domains
        self.domains = sorted(df.DOMAIN.unique())
        # Dictionary of related (ordered) intents grouped by domain
        self.intents_dict = {}
        for intent in df.DOMAIN.unique():
            self.intents_dict[intent] = sorted(df[df.DOMAIN == intent].INTENT.unique())

        if translation_dict is not None:
            self.translation_dict = translation_dict
        else:
            # Build translation dictionary
            self.translation_dict = {}
            for domain, intent_list in self.intents_dict.items():
                for intent in intent_list:
                    self.translation_dict[intent] = self.domains.index(domain) * self.num_max_intents_per_domain + \
                                                    self.intents_dict[domain].index(intent) + 1

        logger.info("Intent translation dictionary: {}".format(self.translation_dict))
        # print("Intents dict:")  # Debug
        # pprint.pprint(self.intents_dict)  # Debug
        # print("Intents translation dict:")  # Debug
        # pprint.pprint(self.translation_dict)  # Debug

    def encode_intent(self, intent_name: str):
        """
        Assigns a numeric value to an intent name
        """
        return self.translation_dict.get(intent_name, 0)


# TEST
# TODO: move to unitary test
if __name__ == "__main__":
    file_path = "data/es/KPI/ES-RECOGNIZER-NLP-201812.csv"
    output_path = "/tmp/mhome-v2-01_formatted.csv"

    formatter = RecognizerLogsFormatter()
    df = formatter.format_logs(file_path, output_path=output_path, output_format="csv", run_in_notebook=False)

    print(df)

