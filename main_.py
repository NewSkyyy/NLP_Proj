import string
import re
import json
import pymorphy2
import math
import _pickle as pickle
import csv
import time
from pathlib import Path
from collections import Counter

RAW_DATA = '1_RawData'
DATA_SET = '2_DataSet'
RESULT = '3_Result'
POS_FILTER = ['VERB', 'NOUN','ADJF','ADJS','COMP','INFN','PRTF','PRTS','GRND','NUMR','ADVB','NPRO','PRED','PREP','CONJ','PRCL','INTJ']
LEMMATIZE_TEXT = True
LINE_NUMBER_FOR_EXTRACTION = 1000 #!!! Количество строк извлекаемых из исходных текстов !!!
WORD_USAGE_THRESHOLD = 0.2; #!!! Порог встречаемости слова в классе, для добавления слова в словарь !!!
CSV_DELIMITER_OPTION = [";","sep=;\n"] #Указание исользуемого разделителя для CSV файла
USE_CSV_DELIMITER = False #Опция использования разделителя CSV файла
GENERATE_READABLE_SOURCE_LIST = False
TO_REMOVE_PATHS = [Path(RESULT, "selected_tf-idf_dict.pickle"), Path(RESULT, "all_groups_external_tf-idf.json"), ]
MORPH = pymorphy2.MorphAnalyzer()

def text_processing(text, lemmatize_text = LEMMATIZE_TEXT):
    if lemmatize_text == True:
        processed_text = lemmatize_sentence(text)
    else:
        unpunctuated_text = text.replace("\n"," ").translate(str.maketrans("", "", TEXT_PUNCTUATION)).strip()
        processed_text = re.findall(r'\w+', unpunctuated_text)
    return Counter(processed_text)

def group_processing(folder_name):
    for group_path in Path(folder_name).iterdir():
        group_dict = {}
        group_uniqueness_dict = {}
        iterator = 1 #Для вывода нумерации
        word_in_class_number = 0
        if not Path(DATA_SET,group_path.name).exists():
            Path(DATA_SET,group_path.name).mkdir(parents = True, exist_ok = True)
        print('{0}\nОбработка текста в дирректории {1}\n{0}'.format(len(list(Path(group_path).name))*'----',Path(group_path).name))
        for text_file in Path(group_path).glob("*.txt"):
            try:
                file_path = Path(text_file)
                file_name = str(file_path.with_suffix("").relative_to(group_path))
                if not Path(DATA_SET, group_path.name, file_name+".pickle").exists():
                    file_dict = dict(text_processing(file_to_text(file_path)))
                    with open(Path(DATA_SET, group_path.name, file_name+".pickle"), 'wb') as file:
                        pickle.dump(file_dict,file)
                else:
                    with open(Path(DATA_SET, group_path.name, file_name+".pickle"), 'rb') as file:
                        file_dict = pickle.load(file)
                word_in_text_number = 0
                for word, value in file_dict.items():
                    if word in group_dict.keys():
                        group_dict.update({word:group_dict.get(word)+value})
                        group_uniqueness_dict.update({word:group_uniqueness_dict.get(word)+1})
                    else:
                        group_dict.update({word:value})
                        group_uniqueness_dict.update({word:1})
                        word_in_class_number+=1
                    word_in_text_number +=1
                print('{0}. {1} \n |Уникальных слов в тексте: {2}|'.format(iterator, file_name, word_in_text_number))
                
            except:
                file_path = Path(text_file)
                file_name = str(file_path.with_suffix("").relative_to(group_path))
                print("{0}. {1}\n Ошибка обработки текста".format(iterator, file_name))
                pass
            iterator += 1
        group_uniqueness_dict.update({word:(value/(iterator - 1)) for word, value in group_uniqueness_dict.items()})
        with open(Path(DATA_SET, group_path.name, "group_word_count_dict.pickle"), 'wb') as file:
           pickle.dump(dict(group_dict),file)
        with open(Path(DATA_SET, group_path.name, "group_uniqueness_dict.pickle"), 'wb') as file:
           pickle.dump(dict(group_uniqueness_dict),file)
        print('\n|Уникальных слов в дирректории "{0}": {1}|\n{2}\n{2}\n'.format(Path(group_path).name, word_in_class_number, 60 * '-'))
    return 0

def group_tf_dict():
    number_of_groups = len(list(Path(DATA_SET).iterdir()))
    all_groups_dict = {}
    result_dict = {}
    idf_groups_dict = {}
    for group_folder in Path(DATA_SET).iterdir():
        read_counted_dict_path = Path(group_folder,"group_word_count_dict.pickle")
        read_pattern_dict_path = Path(group_folder,"group_uniqueness_dict.pickle")
        write_file_path = Path(group_folder,"group_word_frequency_dict.pickle")
        with open(read_counted_dict_path, 'rb') as file:
            counted_dict = pickle.load(file)
        with open(read_pattern_dict_path, 'rb') as file:
            pattern_dict = pickle.load(file)
        group_dict = cut_dict(counted_dict, pattern_dict)
        group_frequency_dictionary = word_frequency(group_dict)
        with open(write_file_path, 'wb',) as file:
            pickle.dump(dict(group_frequency_dictionary), file)
        for word, value in group_dict.items():
            if word not in idf_groups_dict.keys():
                idf_groups_dict.update({word: 1})
            else:
                idf_groups_dict.update({word:idf_groups_dict.get(word)+1})
    group_frequencies = []
    for group_folder in Path(DATA_SET).iterdir():
        frequency_file_path = Path(group_folder,"group_word_frequency_dict.pickle")
        with open(frequency_file_path, 'rb') as file:
            group_frequencies.append(dict(pickle.load(file)))
    for word, value in idf_groups_dict.items():
        counter = 0
        value_list = [0] * number_of_groups
        for group in group_frequencies:
            if word not in group.keys():
                value_list[counter] = 0
            else:
                value_list[counter] = group[word] * math.log(number_of_groups/idf_groups_dict[word], number_of_groups)
            all_groups_dict.update({word:value_list})
            counter += 1
    for word, value in all_groups_dict.items():
        if not all_groups_dict[word] == [0] * number_of_groups:
            result_dict.update({word:value})
    if not Path(RESULT).exists():
        Path(RESULT).mkdir(parents = True, exist_ok = True)
    result_path = Path(RESULT, "all_groups_external_tf-idf.json")
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(result_dict, file, ensure_ascii=False, indent = 4)
    return 0

def file_to_text(file_path):
    try:
        try:
            file = open(file_path, 'r', encoding = "utf-8")
            text = file.read()
            file.close()
        except:
            file = open(file_path, 'r', encoding = "cp1251")
            text = file.read()
            file.close()
    except:
        raise Exception("Неподдерживаемая кодировка текстового файла: " + file_path)
    return text.lower()

def lemmatize_sentence(text):
    lemmatized_text = []
    text_to_list = re.findall(r'\w*', text)
    for word in text_to_list:
        morphed_word = MORPH.parse(word)[0]
        if (morphed_word.tag.POS in POS_FILTER) or ('LATN' in morphed_word.tag): # https://pymorphy2.readthedocs.io/en/stable/user/grammemes.html#grammeme-docs
            lemmatized_text.append(morphed_word.normal_form)
    return lemmatized_text
    
def word_frequency(counted_text):
    total_value = sum(counted_text.values())
    frequency_dictionary = {word: value/ total_value for word, value in counted_text.items()}
    return frequency_dictionary

def cut_dict(orig_dict, pattern_dict,threshold = WORD_USAGE_THRESHOLD):
    result_dict = {}
    for word, value in orig_dict.items():
        if pattern_dict.get(word) >= threshold:
            result_dict.update({word: value})
    return result_dict
    
def result_to_csv():
    source_path = Path(RESULT, "all_groups_external_tf-idf.json")
    listed_dict = []
    with open(source_path, 'r', encoding='utf-8') as file:
        word_dict = json.load(file)
    for i in range(len(list(Path(RAW_DATA).iterdir()))):
        word_group_value = []
        for word, value in word_dict.items():
            word_group_value.append([word, value[i]])
        word_group_value = sorted(word_group_value, key = lambda x:x[1],reverse = True)
        listed_dict.append(word_group_value)
    selected_word_dict(listed_dict, word_dict) # Процесс отсеивания первых n строк из таблицы
    listed_dict = columns_to_rows(listed_dict)
    result_path = Path(RESULT, "result.csv")
    table_header = []
    for folder_name in list(Path(RAW_DATA).iterdir()):
        table_header += [f'Слова из каталога {folder_name}', "TF-IDF"]
    with open(result_path, 'w', newline='') as csvfile:
        csvfile.write(CSV_DELIMITER_OPTION[1])
        writer = csv.writer(csvfile, delimiter= CSV_DELIMITER_OPTION[0])
        writer.writerow(table_header)
        for row in listed_dict:
            writer.writerow(row)
    return 0
        
def columns_to_rows(original_list):
    result_list = []
    for i in range(len(original_list[0])):
        new_row = []
        for sub_list in original_list:
            new_row.extend(sub_list[i])
        result_list.append(new_row)
    return result_list

def readable_dictionary_sources():
    dict_path = Path(RESULT, "selected_tf-idf_dict.pickle")
    sources_dict = {}
    possible_sources_paths = list(Path(DATA_SET).iterdir())
    with open(dict_path, 'rb') as file:
        words_dict = pickle.load(file)
    for word, value in words_dict.items():
        word_sources = []
        for value in words_dict[word]:
            word_value_sources = []
            if value:
                possible_sources = words_dict[word].index(value)
                for dict_file in Path(possible_sources_paths[possible_sources]).glob("*[!dict].pickle"):
                    dict_file_path = Path(dict_file)
                    with open(dict_file_path, 'rb') as file:
                        possible_source = pickle.load(file)
                        if word in possible_source:
                            word_value_sources.append([Path(dict_file).stem, file_word_frequency(word, possible_source)])
            word_sources.append(word_value_sources)
        sources_dict.update({word:word_sources})
    result_path = Path(RESULT, "readable_dict_sources.json")
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(sources_dict, file, ensure_ascii=False, indent = 4)
    return 0        


def dictionary_sources():
    dict_path = Path(RESULT, "selected_tf-idf_dict.pickle")
    sources_dict = {}
    possible_sources_paths = list(Path(DATA_SET).iterdir())
    with open(dict_path, 'rb') as file:
        words_dict = pickle.load(file)
    for sources_folder in possible_sources_paths:
        for source in Path(sources_folder).glob("*[!dict].pickle"):
            with open(source, 'rb') as file:
                possible_source = pickle.load(file)
            if (possible_source.keys() & words_dict.keys()):
                source_keys = []
                for key in list(possible_source.keys() & words_dict.keys()):
                    source_keys.append([key, file_word_frequency(key, possible_source)])
            sources_dict.update({Path(source).stem:source_keys})
    result_path = Path(RESULT, "dict_sources.json")
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(sources_dict, file, ensure_ascii=False, indent = 4)
    return 0 

def final_constructor(use_certain_delimeter = USE_CSV_DELIMITER):
    sources_paths = list(Path(RAW_DATA).iterdir())
    sources_dict_path = Path(RESULT, "dict_sources.json")
    with open(sources_dict_path, 'r', encoding='utf-8') as file:
        sources_dict = json.load(file)
    selected_tf_idf_dict_path = Path(RESULT, "selected_tf-idf_dict.pickle")
    with open(selected_tf_idf_dict_path, 'rb') as file:
        selected_tf_idf_dict = pickle.load(file)
    words_list = [0]*2 + list(selected_tf_idf_dict.keys())
    result_path = Path(RESULT, "formatted_result.csv")
    table_header = ["Группа", "Имя Файла", "Количество слов", "Количество предложений"]
    table_header.extend(words_list[2:])
    with open(result_path, 'w', newline='') as csvfile:
        if use_certain_delimeter:
            csvfile.write(CSV_DELIMITER_OPTION[1])
        writer = csv.writer(csvfile, delimiter= CSV_DELIMITER_OPTION[0])
        writer.writerow(table_header)
        group_counter = 0
        for sources_group in sources_paths:
            for text_file in Path(sources_group).glob("*.txt"):
                if Path(text_file).stem in sources_dict:
                    row_content = [Path(sources_group).name, Path(text_file).stem]
                    words_frequencies = [0] * (len(selected_tf_idf_dict.keys())+2)
                    source_words = sources_dict.get(Path(text_file).stem)
                    for word, value in source_words:
                        word_index = words_list.index(word)
                        words_frequencies[word_index] = value
                    row_content.extend(words_frequencies)
                    writer.writerow(row_content)
            group_counter+=1
    return 0
    

def string_formating(template):
    return eval(f"f'{template}'")

def file_word_frequency(word, file_dict):
    frequency = file_dict.get(word)/sum(file_dict.values())
    return frequency

def selected_word_dict(result_list, source_dict, number_for_extraction = LINE_NUMBER_FOR_EXTRACTION):
    selected_words = {}
    extracted_words = set()
    if (len(result_list[0]) > number_for_extraction):
        extract_count = number_for_extraction
    else:
        extract_count = len(result_list[0])
    for word_list in result_list:
        for i in range(extract_count):
            extracted_words.add(word_list[i][0])
    for word in list(extracted_words):
        selected_words.update({word:source_dict.get(word)})
    with open(Path(RESULT,"selected_tf-idf_dict.pickle"), 'wb',) as file:
            pickle.dump(dict(selected_words), file)
    return 0

def garbage_removal(to_remove = TO_REMOVE_PATHS):
    for path in to_remove:
        if (path):
            path.unlink()
    return 0
    
# Код реализует алгоритм извлечения наиболее значимых слов из текстов группированых каталогов, 
# размещенных в каталоге 1_RawData.
# Пример:
# В каталоге 1_RawData содержится каталоги с названиями Пушкин, Гоголь, Толстой, 
# в которых содержатся текстовые файлы, с текстами произведений писателей из названий каталогов.
# В результирующем каталоге 3_Result, в результате работы программы, будут содрежатся:
# json файл dict_sources, сожержащий слова, содержащиеся в тексте и их частоты относительно текста
# csv файл result, содержащий слова с наибольшим модифицированным TF-IDF (наиболее значимые) и изначение TF-IDF
# csv файл formatted_result, содержащий первые n извлекаемых слов из таблицы result, с указанием их принадлежности к каталогу и их внутритекстовых частот.
# В каталоге 2_DataSet содержатся промежуточные файлы, необходимые для расчетов.
# Необходимые для работы файлы: каталог 1_RawData с каталогами с различными именами (> 1 каталога), содержащие текстовые файлы с текстами.

def main():
    start_time = time.time() #Старт расчета затраченного времени
    group_processing("1_RawData") #Сбор слов и их колличества из исходных текстов
    group_tf_dict() #Расчет межклассовых метрик для собранных слов
    result_to_csv()#Перевод json-файла в табличный вид
    dictionary_sources()#Генерация словаря, содержащего частоты выбранных слов относительно источников
    final_constructor()#Сборка данных в обрабатываемый формат
    if (GENERATE_READABLE_SOURCE_LIST):
        readable_dictionary_sources() #Формирование читаемого варианта источников слов с частотами
    garbage_removal()#Удаление ненужных файлов
    print("--- Program works for %s seconds ---" % (time.time() - start_time)) #Окончание расчета затраченного времени
    return 0
    
main()